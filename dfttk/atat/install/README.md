# Installation
- Download the ATAT source code from https://www.brown.edu/Departments/Engineering/Labs/avdw/atat/.

- Unzip the package

        tar -xvf atat*.tar.gz

- Copy icamag.c++, fxvector.h, and makefile from this folder into atat/src. Then navigate to the atat/src folder and type:

        make icamag

- Navigate to the atat folder and type:

        make
        make install

- Navigate to the bashrc file using:

        vi ~/.bashrc

- Add to PATH:

        export PATH=$PATH:/path/to/ATAT/bin/

-Implement the changes:

        source ~/.bashrc
