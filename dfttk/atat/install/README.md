# Installation
1. Download the ATAT source code from https://www.brown.edu/Departments/Engineering/Labs/avdw/atat/. If ATAT is already installed skip to step 3.

2. Unzip the package

        tar -xvf atat*.tar.gz

3. Replace the fxvector.h and makefile in atat/src with the fxvector.h and makefile from this folder.

4. Copy icamag.c++ from this folder into atat/src. Then navigate to the atat/src folder and type:

        make icamag

5. Navigate to the atat folder and type:

        make
        make install

6. Navigate to the bashrc file using:

        vi ~/.bashrc

7. Add to PATH:

        export PATH=$PATH:/path/to/ATAT/bin/

8. Implement the changes:

        source ~/.bashrc
