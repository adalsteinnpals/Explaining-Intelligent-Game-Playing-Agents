from utils import get_FenBatchProvider

def test_get_FenBatchProvider():
    batch_size = 100
    train_loader = get_FenBatchProvider(batch_size=batch_size)

    for i in range(10):
        batch = next(train_loader)
        print(batch)


if __name__ == '__main__':
    test_get_FenBatchProvider()