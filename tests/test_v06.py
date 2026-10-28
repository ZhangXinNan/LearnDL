"""v0.6 (Steps 47-51) offline checks for classification, loaders, and MNIST parsing.

Uses a tiny synthetic MNIST IDX fixture: this is NOT a real MNIST accuracy benchmark.
Run from repository root: python -m unittest discover -s tests -p 'test_v06.py' -v
"""
import gzip
import os
import tempfile
import unittest
from unittest import mock
import numpy as np

import dezero
from dezero import Variable
from dezero.dataloaders import DataLoader
from dezero.datasets import Dataset, Spiral, MNIST
from dezero.models import MLP
from dezero.optimizers import SGD
import dezero.functions as F


class ArrayDataset(Dataset):
    def prepare(self):
        self.data = np.arange(13, dtype=np.float32).reshape(-1, 1)
        self.label = np.arange(13, dtype=np.int64) % 3


class V06Tests(unittest.TestCase):
    def test_softmax_is_stable_and_gradient_is_correct(self):
        xdata = np.array([[1000., 999., 998.], [-1000., 20., 30.]])
        x = Variable(xdata.copy())
        y = F.softmax(x)
        np.testing.assert_allclose(y.data.sum(axis=1), [1., 1.], atol=1e-14)
        self.assertTrue(np.isfinite(y.data).all())
        weights = np.array([[1., 2., 3.], [4., 2., 1.]])
        F.sum(y * weights).backward()
        numeric = np.empty_like(xdata)
        eps = 1e-5
        for ix in np.ndindex(xdata.shape):
            plus, minus = xdata.copy(), xdata.copy()
            plus[ix] += eps
            minus[ix] -= eps
            numeric[ix] = (float((F.softmax(Variable(plus)).data * weights).sum())
                           - float((F.softmax(Variable(minus)).data * weights).sum())) / (2*eps)
        np.testing.assert_allclose(x.grad.data, numeric, atol=2e-8, rtol=1e-4)

    def test_cross_entropy_gradient_and_column_targets(self):
        scores = np.array([[3., 1., -2.], [0., -1., 2.]], dtype=np.float64)
        labels = np.array([0, 2], dtype=np.int64)
        def check(t):
            x = Variable(scores.copy())
            loss = F.softmax_cross_entropy(x, t)
            self.assertEqual(loss.shape, ())
            self.assertTrue(np.isfinite(loss.data))
            loss.backward()
            self.assertEqual(x.grad.shape, x.shape)
            eps = 1e-5
            numeric = np.empty_like(scores)
            for idx in np.ndindex(scores.shape):
                plus, minus = scores.copy(), scores.copy()
                plus[idx] += eps
                minus[idx] -= eps
                numeric[idx] = (float(F.softmax_cross_entropy(Variable(plus), t).data)
                                - float(F.softmax_cross_entropy(Variable(minus), t).data)) / (2*eps)
            np.testing.assert_allclose(x.grad.data, numeric, rtol=1e-5, atol=1e-7)
        check(labels)
        check(labels.reshape(-1, 1))

    def test_dataloader_complete_and_partial_batches(self):
        dataset = ArrayDataset()
        loader = DataLoader(dataset, batch_size=5, shuffle=False)
        sizes, visited = [], []
        for x, t in loader:
            sizes.append(len(t))
            visited.extend(int(a) for a in x[:, 0])
        self.assertEqual(sizes, [5, 5, 3])
        self.assertEqual(visited, list(range(13)))
        # The loader automatically resets at exhaustion.
        self.assertEqual(next(iter(loader))[0][:, 0].tolist(), list(range(5)))
        np.random.seed(123)
        shuffled = DataLoader(dataset, batch_size=5, shuffle=True)
        visited = [int(v) for x, _ in shuffled for v in x[:, 0]]
        self.assertEqual(sorted(visited), list(range(13)))
        self.assertNotEqual(visited, list(range(13)))

    def test_get_item_scatter_add_and_second_derivative(self):
        x = Variable(np.array([1., 2., 3.]))
        F.sum(x[[1, 1, 0]]).backward()
        np.testing.assert_array_equal(x.grad.data, [1., 2., 0.])

        x = Variable(np.array([1., 2., 3.]))
        F.sum(x[[0, 2]] ** 2).backward(create_graph=True)
        first = x.grad
        np.testing.assert_allclose(first.data, [2., 0., 6.])
        x.cleargrad()
        F.sum(first).backward()
        np.testing.assert_allclose(x.grad.data, [2., 0., 2.])

    def test_accuracy_and_no_grad_evaluation(self):
        y = Variable(np.array([[0., 1., 2.], [5., 1., 0.], [0., 3., 1.]]))
        labels = np.array([2, 0, 0])
        acc = F.accuracy(y, labels)
        self.assertAlmostEqual(float(acc.data), 2/3)
        with dezero.no_grad():
            scores = F.softmax(y)
        self.assertIsNone(scores.creator)

    def test_mnist_idx_reader_and_loader_without_network(self):
        n = 4
        images = np.arange(n*28*28, dtype=np.uint8).reshape(n, 28, 28)
        labels = np.array([3, 1, 4, 2], dtype=np.uint8)
        with tempfile.TemporaryDirectory() as d:
            image_path = os.path.join(d, 'images.gz')
            label_path = os.path.join(d, 'labels.gz')
            with gzip.open(image_path, 'wb') as f:
                f.write(b'\x00\x00\x08\x03' + n.to_bytes(4, 'big')
                        + (28).to_bytes(4, 'big')*2 + images.tobytes())
            with gzip.open(label_path, 'wb') as f:
                f.write(b'\x00\x00\x08\x01' + n.to_bytes(4, 'big') + labels.tobytes())
            def no_download(url):
                return image_path if 'images' in url else label_path
            with mock.patch('dezero.datasets.get_file', side_effect=no_download):
                train, test = MNIST(train=True), MNIST(train=False)
            for ds in (train, test):
                self.assertEqual(len(ds), n)
                sample, label = ds[0]
                self.assertEqual(sample.shape, (784,))
                self.assertEqual(sample.dtype, np.dtype('float32'))
                self.assertTrue(np.all(sample >= 0) and np.all(sample <= 1))
                self.assertEqual(int(label), 3)
                batch_x, batch_t = next(iter(DataLoader(ds, batch_size=3)))
                self.assertEqual(batch_x.shape, (3, 784))
                self.assertEqual(batch_t.shape, (3,))

    def test_spiral_mlp_training_reduces_loss(self):
        np.random.seed(21)
        train = Spiral(train=True)
        # Spiral sets its own RNG seed. Fix initialization for reproducibility.
        np.random.seed(3)
        model = MLP((10, 3))
        optimizer = SGD(1.).setup(model)
        loader = DataLoader(train, batch_size=30)
        losses = []
        for _ in range(110):
            total = 0
            for x, t in loader:
                y = model(x)
                loss = F.softmax_cross_entropy(y, t)
                model.cleargrads()
                loss.backward()
                optimizer.update()
                total += float(loss.data)*len(t)
            losses.append(total/len(train))
        self.assertLess(losses[-1], .65)
        self.assertLess(losses[-1], .75*losses[0])
        with dezero.no_grad():
            test = Spiral(train=False)
            correct = sum(float(F.accuracy(model(x), t).data)*len(t)
                          for x, t in DataLoader(test, batch_size=50, shuffle=False))
        self.assertGreater(correct/len(test), .6)


if __name__ == '__main__':
    unittest.main()
