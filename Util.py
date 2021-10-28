

def debug(*args, sep=' ', end='\n', file=None):
    print(*args, sep=sep, end=end, file=file)



if __name__ == '__main__':
    debug("hello", 123, sep="\n")