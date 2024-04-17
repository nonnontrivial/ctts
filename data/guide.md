# guide

the first step of training the model is unpacking the tarball which contains the
csv files from the public globe at night monitoring network.

the tarball is not included in this repo due to size constraints, but the files
are available [on the gan mn site](https://globeatnight.org/gan-mn/).

## building the tarball

- download \*.csv files you want to include in training data from the above link
  and store them in a directory called `gan_mn`
- create a tarball with `tar -czvf gan_mn.tar.gz ./path/to/gan_mn`

## running the unpack script

```py
cd ctts
python -m data.unpack
```

## sources

### globe at night monitoring network

- https://globeatnight.org/gan-mn/

### artificial night sky brightness

- https://djlorenz.github.io/astronomy/lp2022/
