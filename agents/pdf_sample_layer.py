import numpy as np
import torch


def pdf_sample(pdf, uniform_noise):
    pdf = pdf / (torch.sum(pdf, dim=1, keepdim=True) + 1e-36)
    cdf = torch.cumsum(pdf, dim=1) - pdf
    indices = torch.sum(torch.less(cdf, uniform_noise).to(torch.int32), dim=1) - 1
    return indices


def test2():
    batch_size = 1024
    n = 3

    counter = [0. for _ in range(n)]
    for i in range(1000):
        pdf_batch = torch.tensor([[2.0 ** i for i in range(1, n + 1)] for _ in range(batch_size)], dtype=torch.float32)
        noise = torch.tensor(np.random.rand(batch_size, 1), dtype=torch.float32)
        indices_out = pdf_sample(pdf_batch, noise)
        for i in indices_out:
            counter[indices_out[i]] += 1

    for i in range(n):
        print(counter[i] * 1.0 / 100 / batch_size)


if __name__ == '__main__':
    np.random.seed(0)
    test2()

