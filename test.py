class Test():
    def __init__(self):
        self.private_fn()

    def _private_fn(self):
        print('private')


if __name__ == '__main__':
    test = Test()

