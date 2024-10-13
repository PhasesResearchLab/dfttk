.SUFFIXES: .hlp .c++ .hh
BINDIR=$(HOME)/bin/

#for g++ compiler on all platforms
CXX=g++ 

#for MPI: limited implementation at this point
#CXX=mpiCC -DATAT_MPI 

#note that BINDIR and CXX are overridden if called by ../makefile

LDLIBS=-lm
#if you want to use blas/lapack
#LDLIBS=-lm -lgomp -lmkl_intel_lp64 -lmkl_gnu_thread -lmkl_core
#note that -O can cause problems...
include makefileflags
#CXXFLAGS=$(PATCHCXXFLAGS) 
#CXXFLAGS=$(PATCHCXXFLAGS) -O3 -DATATUSELAPACK 
#CXXFLAGS=$(PATCHCXXFLAGS) -DDEGE_GS 
#CXXFLAGS=$(PATCHCXXFLAGS) -DSLOWENUMALGO
#CXXFLAGS=$(PATCHCXXFLAGS) -DDEGE_GS -DSLOWENUMALGO
#CXXFLAGS=$(PATCHCXXFLAGS) -DDEBUG -g -DSLOWENUMALGO #-DDEGE_GS
#CXXFLAGS=$(PATCHCXXFLAGS) -DDEBUG -g #-DDEGE_GS 
CXXFLAGS=$(PATCHCXXFLAGS) -O3 
#CXXFLAGS=$(PATCHCXXFLAGS) -O3 -DQUICKNEIGHBORALGO 

#for native compiler on SGI
#need to include ./patchstrstr in ./patchlang 
#CXX=CC
#LDLIBS=-lm
#CXXFLAGS=-LANG: ansi-for-init-scope=ON

#for native compiler on IBM SP
#need to include ./patchagg and ./patchfor in ./patchlang
#CXX=xlC
#LDLIBS=-lm -lcomplex
#CXXFLAGS=-DOLD_COMPLEX -DSTRING_FIX

CC=${CXX}
CFLAGS=$(CXXFLAGS)

ALLSCRIPT=getclus mapsrep mmapsrep str2xyz makelat clusterexpand mkteci clusterpredict checkrelax calcbulk mcdroplast str2cif wycked getproto sqs2tdb symbrklib
ALLPRG=maps mmaps emc2 phb checkcell corrdump kmesh genstr gensqs mcsqs nntouch fixcell csfit cv cellcvrt lsfit fitsvsl svsl felec pdef fitfc nnshell memc2 gce gencs triph strpath infdet apb icamag
ALLH=anyfft.h drawpd.h gstate.h lstsqr.h multipoly.h stringo.h array.h equil.h integer.h machdep.h parse.h teci.h arraylist.h fftn.h keci.h kmeci.h mclib.h phonlib.h vectmac.h calccorr.h findsym.h lattype.h misc.h plugin.h version.h calcmf.h fixagg.h linalg.h mmclib.h predrs.h xtalutil.h chull.h fxvector.h linklist.h mrefine.h refine.h clus_str.h getvalue.h linsolve.h mteci.h ridge.h kspacecs.cc tensor.h gceutil.h tensorsym.h binstream.h mpiinterf.h normal.h apb.h opti.h strinterf.h meshutil.h

PLUGINMAPS=predrs.o predcs.o tlambda.o ridge.o
PLUGINMMAPS=predes.o
PLUGINEMC2=kspacecs.o
PLUGINMEMC2=kspacees.o
PLUGINSVSL=morsepot.o morsepotc3.o
PLUGINCORR=

all:	$(ALLPRG)

#make sure to include -DSLOWENUMALGO if compiling mpi version
mpi:	mpigenstr mpimmaps splitjob dispatch

headers.o: ${ALLH}

maps: headers.o stringo.o parse.o xtalutil.o integer.o findsym.o lattype.o calccorr.o refine.o clus_str.o linalg.o lstsqr.o gstate.o linsolve.o getvalue.o mapshelp.o $(PLUGINMAPS)
mmaps: headers.o stringo.o parse.o xtalutil.o integer.o findsym.o lattype.o calccorr.o mrefine.o clus_str.o linalg.o lstsqr.o linsolve.o getvalue.o mmapshelp.o chull.o mpiinterf.o $(PLUGINMMAPS) $(PLUGINCORR)
emc2: headers.o teci.o stringo.o parse.o xtalutil.o integer.o findsym.o clus_str.o calccorr.o getvalue.o lstsqr.o linalg.o linsolve.o drawpd.o mclib.o keci.o emc2help.o anyfft.o fftn.o $(PLUGINEMC2)
phb: headers.o teci.o stringo.o parse.o xtalutil.o integer.o findsym.o clus_str.o calccorr.o getvalue.o lstsqr.o linalg.o linsolve.o drawpd.o mclib.o keci.o phbhelp.o anyfft.o fftn.o $(PLUGINEMC2)
checkcell: headers.o parse.o xtalutil.o integer.o stringo.o getvalue.o
analrelax: headers.o parse.o xtalutil.o integer.o stringo.o getvalue.o linsolve.o linalg.o
corrdump: headers.o stringo.o parse.o xtalutil.o integer.o findsym.o calccorr.o clus_str.o getvalue.o corrdumphelp.o linalg.o linsolve.o lstsqr.o $(PLUGINCORR)
skelread: headers.o stringo.o parse.o xtalutil.o integer.o findsym.o calccorr.o clus_str.o getvalue.o linalg.o linsolve.o lstsqr.o
gce: headers.o stringo.o parse.o xtalutil.o integer.o findsym.o calccorr.o clus_str.o getvalue.o gcehelp.o linalg.o linsolve.o lstsqr.o gceutil.o tensorsym.o phonlib.o multipoly.o $(PLUGINCORR)
viewgce: headers.o headers.o stringo.o parse.o xtalutil.o integer.o findsym.o calccorr.o clus_str.o getvalue.o corrdumphelp.o linalg.o linsolve.o lstsqr.o gceutil.o tensorsym.o phonlib.o multipoly.o
gencs: headers.o stringo.o parse.o xtalutil.o integer.o findsym.o getvalue.o linalg.o linsolve.o lstsqr.o tensorsym.o phonlib.o multipoly.o

kmesh: headers.o findsym.o integer.o xtalutil.o getvalue.o stringo.o
genstr: headers.o stringo.o parse.o xtalutil.o integer.o findsym.o clus_str.o getvalue.o lattype.o mpiinterf.o
gensqs: headers.o stringo.o parse.o xtalutil.o integer.o findsym.o clus_str.o getvalue.o lattype.o calccorr.o gensqshelp.o linalg.o linsolve.o lstsqr.o $(PLUGINCORR)
mcsqs: headers.o stringo.o parse.o xtalutil.o integer.o findsym.o clus_str.o getvalue.o lattype.o calccorr.o mcsqshelp.o linalg.o linsolve.o lstsqr.o $(PLUGINCORR)
nntouch: headers.o stringo.o parse.o xtalutil.o integer.o getvalue.o 
cellcvrt: headers.o stringo.o parse.o xtalutil.o integer.o getvalue.o findsym.o lattype.o
calces: headers.o stringo.o parse.o xtalutil.o integer.o getvalue.o
fixcell: headers.o lattype.o parse.o integer.o xtalutil.o stringo.o getvalue.o
csfit: headers.o xtalutil.o parse.o getvalue.o lstsqr.o stringo.o integer.o linalg.o linsolve.o predcshelp.o
cv: headers.o linalg.o lstsqr.o linsolve.o integer.o getvalue.o cvhelp.o stringo.o
lsfit: headers.o lstsqr.o linsolve.o getvalue.o linalg.o integer.o ridge.o stringo.o
svsl: headers.o xtalutil.o linalg.o integer.o lstsqr.o getvalue.o findsym.o linsolve.o parse.o stringo.o phonlib.o svslhelp.o calccorr.o clus_str.o multipoly.o $(PLUGINSVSL)
fitsvsl: headers.o xtalutil.o linalg.o integer.o lstsqr.o getvalue.o findsym.o linsolve.o parse.o stringo.o phonlib.o lattype.o fitsvslhelp.o multipoly.o calccorr.o clus_str.o
pdef: headers.o xtalutil.o linalg.o integer.o lstsqr.o getvalue.o findsym.o linsolve.o parse.o stringo.o lattype.o
nnshell: headers.o xtalutil.o parse.o integer.o stringo.o getvalue.o
felec: headers.o getvalue.o integer.o stringo.o felechelp.o parse.o xtalutil.o
fempmag: headers.o getvalue.o integer.o stringo.o fempmaghelp.o parse.o xtalutil.o
fmag: headers.o getvalue.o integer.o stringo.o xtalutil.o parse.o
fitfc: headers.o xtalutil.o linalg.o integer.o lstsqr.o getvalue.o findsym.o linsolve.o parse.o stringo.o phonlib.o lattype.o fitfchelp.o multipoly.o
memc2: headers.o mteci.o stringo.o parse.o xtalutil.o integer.o findsym.o clus_str.o calccorr.o getvalue.o lstsqr.o linalg.o linsolve.o mmclib.o equil.o calcmf.o memc2help.o multipoly.o chull.o anyfft.o fftn.o kmeci.o $(PLUGINMEMC2) $(PLUGINCORR)
calcelas: headers.o xtalutil.o linalg.o integer.o lstsqr.o getvalue.o findsym.o linsolve.o parse.o stringo.o tensorsym.o phonlib.o multipoly.o
strainder: headers.o xtalutil.o linalg.o integer.o lstsqr.o getvalue.o findsym.o linsolve.o parse.o stringo.o tensorsym.o phonlib.o multipoly.o
calchull: headers.o linalg.o integer.o lstsqr.o getvalue.o linsolve.o chull.o stringo.o
strpath: headers.o normal.o xtalutil.o integer.o parse.o stringo.o getvalue.o
apb: headers.o xtalutil.o integer.o parse.o stringo.o getvalue.o apbhelp.o
infdet: headers.o opti.o strinterf.o integer.o linalg.o linsolve.o parse.o stringo.o xtalutil.o getvalue.o infdethelp.o
#experimental
epic: headers.o opti.o strinterf.o integer.o linalg.o linsolve.o parse.o stringo.o xtalutil.o getvalue.o epichelp.o
loctst: headers.o integer.o parse.o stringo.o xtalutil.o getvalue.o
iwulff: headers.o stringo.o parse.o xtalutil.o integer.o getvalue.o
clusover: headers.o linalg.o integer.o linsolve.o
tryclus: headers.o xtalutil.o parse.o integer.o stringo.o getvalue.o
csl: headers.o stringo.o xtalutil.o findsym.o parse.o getvalue.o integer.o
test: headers.o test.o linsolve.o linalg.o chull.o integer.o getvalue.o lstsqr.o stringo.o
stm: headers.o stringo.o xtalutil.o findsym.o parse.o getvalue.o integer.o
cna: headers.o stringo.o xtalutil.o parse.o integer.o getvalue.o
cmpstr: headers.o stringo.o parse.o xtalutil.o integer.o getvalue.o
ehmass: headers.o parse.o xtalutil.o integer.o getvalue.o lstsqr.o linalg.o linsolve.o stringo.o findsym.o
seebeck: headers.o parse.o xtalutil.o integer.o getvalue.o lstsqr.o linalg.o linsolve.o stringo.o findsym.o
bgsdqmc: headers.o normal.o
vctriv: headers.o normal.o xtalutil.o integer.o parse.o stringo.o getvalue.o
hessian: headers.o linalg.o linsolve.o integer.o stringo.o getvalue.o
triph: headers.o stringo.o integer.o getvalue.o
simplexize: headers.o stringo.o integer.o getvalue.o linalg.o linsolve.o meshutil.o
mkaxes: headers.o stringo.o integer.o getvalue.o meshutil.o linalg.o linsolve.o
pathint: headers.o linsolve.o integer.o getvalue.o stringo.o linalg.o
greadyneb: headers.o linsolve.o linalg.o integer.o getvalue.o stringo.o
plotcurv: headers.o linsolve.o linalg.o
mpitest: headers.o mpiinterf.o xtalutil.o parse.o stringo.o integer.o
icamag: headers.o stringo.o parse.o xtalutil.o integer.o findsym.o clus_str.o getvalue.o
#end

splitjob: splitjob.o
dispatch: headers.o mpiinterf.o
mpigenstr: genstr
	cp -f genstr mpigenstr
mpimmaps: mmaps
	cp -f mmaps mpimmaps

.hlp.cc:
	./makehelp.bash < $*.hlp > $*.cc

.c++.cc:
	./patchlang < $*.c++ > $*.cc

.PRECIOUS : %.cc

.hh.h:
	./patchlang < $*.hh > $*.h

install:
	../safecp $(ALLPRG) $(ALLSCRIPT) $(BINDIR)

clean:
	rm -f *.o *.cc *.c *.h *.exe
	rm -f $(ALLPRG)
