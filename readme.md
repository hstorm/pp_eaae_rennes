
# Numpyro Examples EAAE 2023
Contact: *hugo.storm@ilr.uni-bonn.de*




# Setup environment using pipenv

1) Install pipenv dependencies ```pipenv install --dev```

2) In order to use GPU support it is required to install numpyro[cuda] manually using wheels, currently seems to be not supported in pipenv

    ```
    pipenv shell

    pip install numpyro[cuda] -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
    ```
    
    Optionally install flax manually:
    ```pip install flax```