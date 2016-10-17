
CodeMend - Interactive Programming Environment Supported by Bimodal Embedding Model
=============================

Installation
------------

First, download and install `Anaconda` at https://www.continuum.io/downloads. Make sure you get Anaconda with Python 2 (e.g., Python 2.7).

Next, clone `codemend` to your local disk::

    git clone https://github.com/ronxin/codemend.git

Then, install it::


    cd codemend
    python setup.py develop

Dependencies should be automatically resolved. The `develop` option is important, because it creates a symbolic link to the original repository during installation, so any change made to the original files will be reflected immediately.

**Note**: The only reason we require Anaconda is because it offers off-the-shelf installation of `matplotlib` and other `pylab` modules. Installation without Anaconda is not recommended but possible (see http://xymath.readthedocs.org/en/latest/quickstart.html).


Running the demo
----------------

The demo is a Python server. If codemend is properly installed, then the server should be launched from anywhere by typing the following in a terminal::

    codemend

By default, it uses port 9001. So the URL is <http://localhost:9001>.

To use a custom port, e.g., 8888::

The above configurations only allow traffic from the local host (i.e., browsers on the same machine). If you want to allow access from remote clients, use::

    codemend 9001 0.0.0.0

where 9001 can be replaced with any desired port number, and 0.0.0.0 means any IP address is allowed to visit. Note that this is currently very dangerous as this allows anybody to run any python command (including those that delete files on the local system).


**Windows users**: to run codemend demo, you need to type in::

    python -m codemend.demo.start_server

The parameters (port and host address) are the same.

Update Version
--------------

To update to the latest version::

    cd codemend
    git pull
    python setup.py develop
