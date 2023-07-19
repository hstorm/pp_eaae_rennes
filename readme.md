
# Numpyro Examples EAAE 2023
Contact: *hugo.storm@ilr.uni-bonn.de*




# Setup environment using pipenv

1) Install pipenv dependencies ```pipenv install --dev```

2) Install numpyro[cuda] manually using wheels, this does not work in pipenv

    ```pip install numpyro[cuda] -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html```
    
    Additionally, you need to install flax manually:
    ```pip install flax```