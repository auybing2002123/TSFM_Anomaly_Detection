FROM tensorflow/tensorflow:1.15.5-py3

RUN apt-get update \
    && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir \
    git+https://github.com/thu-ml/zhusuan.git \
    git+https://github.com/haowen-xu/tfsnippet.git@v0.2.0-alpha1 \
    six \
    PyYAML \
    numpy \
    pandas \
    ipython \
    imageio \
    fs \
    scikit-learn \
    matplotlib \
    click \
    requests \
    lazy-object-proxy

WORKDIR /workspace/TraceAnomaly
