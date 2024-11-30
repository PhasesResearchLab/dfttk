#include <fstream>
#include "parse.h"
#include "getvalue.h"
#include "version.h"

int main(int argc, char *argv[]) {
  // parsing command line. See getvalue.hh for details;
  char *latfilename="lat.in";
  int sigdig=5;
  int dohelp=0;
  int dummy=0;
  AskStruct options[]={
    {"","Independent Cell Approximation for MAGnetic systems" MAPS_VERSION ", by Axel van de Walle",TITLEVAL,NULL},
    {"-l","Input file defining the lattice (Default: lat.in)",STRINGVAL,&latfilename},
    {"-sig","Number of significant digits printed (Default: 5)",INTVAL,&sigdig},
    {"-d","Use all default values",BOOLVAL,&dummy}
  };
  if (!get_values(argc,argv,countof(options),options)) {
    display_help(countof(options),options);
    return 1;
  }
  
  cout.setf(ios::fixed);
  cout.precision(sigdig);
  
  // parsing lattice and structure files. See parse.hh for detail;
  Structure lat;
  Array<Arrayint> labellookup;
  Array<AutoString> label;
  rMatrix3d axes;
  {
    ifstream latfile(latfilename);
    if (!latfile) ERRORQUIT("Unable to open lattice file");
    parse_lattice_file(&lat.cell, &lat.atom_pos, &lat.atom_type, &labellookup, &label, latfile, &axes);
    wrap_inside_cell(&lat.atom_pos,lat.atom_pos,lat.cell);
  }
  SpaceGroup spacegroup;
  spacegroup.cell=lat.cell;
  find_spacegroup(&spacegroup.point_op,&spacegroup.trans,lat.cell,lat.atom_pos,lat.atom_type);

  Array<int> nb_species(lat.atom_type.get_size());
  for (int i=0; i<nb_species.get_size(); i++) {
    int sz=labellookup(lat.atom_type(i)).get_size();
    if (sz>2) ERRORQUIT("Only up to 2 spin states implemented");
    nb_species(i)=sz;
  }

  LinkedList<Structure> str_list;
  LinkedList<int> mult_list;
  MultiDimIterator<Array<int> > config(nb_species);
  for (; config; config++) {
    Structure str;
    str.cell=lat.cell;
    str.atom_pos=lat.atom_pos;
    str.atom_type=(Array<int> &)config;
    int toadd=1;
    for (int flip=0; flip<2; flip++) {
      LinkedListIterator<Structure> it(str_list);
      LinkedListIterator<int> im(mult_list);
      for (; it; it++,im++) {
	if (equivalent_by_symmetry(*it,str,spacegroup.cell,spacegroup.point_op,spacegroup.trans)) {
	  (*im)++;
	  break;
	}
      }
      if (it) {
	toadd=0;
	break;
      }
      for (int i=0; i<nb_species.get_size(); i++) {
	str.atom_type(i)=nb_species(i)-1-str.atom_type(i);
      }
    }
    if (toadd) {
      str_list << new Structure(str);
      mult_list << new int(1);
    }

  }
  
  LinkedListIterator<Structure> it(str_list);
  LinkedListIterator<int> im(mult_list);
  for (; it; it++,im++) {
    cout << *im << endl;
    write_structure(*it,lat,labellookup,label,axes,cout,0);
    cout << "end" << endl << endl;
  }
}
