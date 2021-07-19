import argparse
from InceptionUNET.predict import predict

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='process image')
    parser.add_argument('filepath', metavar='File Path', type=str, help='Filepath of document to segment')
    args = parser.parse_args()
    out = predict(args.filepath)
