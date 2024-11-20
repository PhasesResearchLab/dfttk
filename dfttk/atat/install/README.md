# Installation
- Download the ATAT source code from https://www.brown.edu/Departments/Engineering/Labs/avdw/atat/.

1. Unzip the package

        tar -xvf atat*.tar.gz

2. Remove the makefile located atat/src.

3. Copy icamag.c++, fxvector.h, and makefile from this folder into atat/src. Then navigate to the atat/src folder and type:

        make icamag

4. Navigate to the atat folder and type:

        make
        make install

5. Navigate to the bashrc file using:

        vi ~/.bashrc

6. Add to PATH:

        export PATH=$PATH:/path/to/ATAT/bin/

6. Implement the changes:

        source ~/.bashrc
