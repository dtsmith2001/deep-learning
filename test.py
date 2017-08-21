import mnist_loader

if __name__ == '__main__':
    train, valid, test = mnist_loader.load_data_wrapper()
    print('Training size {} validation size {} test size {}'.format(train.shape, valid.shape, test.shape))