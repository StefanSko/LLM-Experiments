import os
from modal import Image, Stub


image = (
    Image.debian_slim(python_version="3.11")
    .copy_local_file(local_path='test_file', remote_path='./root')
)

stub = Stub("example-hello-world", image=image)


@stub.function()
def show_dir():
    print(os.getcwd())
    print(os.listdir('.'))
    print('-----')

@stub.local_entrypoint()
def main():
    show_dir.remote()
