#ifndef __FXVECTOR_H__
#define __FXVECTOR_H__

#include <math.h>
#include "misc.h"

#define RESULT_FORALLi(body) \
  FixedVector<T,D> result;\
  for (int i=0; i<D; i++) {\
    body;\
  }\
  return result

#define FORALLi for(int i=0; i<D; i++)

template<class T, int D>
  class FixedVector; 

template<class T, int D>
     int operator==(const FixedVector<T,D>& a, const FixedVector<T,D>& b);
template<class T, int D>
     int operator!=(const FixedVector<T,D>& a, const FixedVector<T,D>& b);
template<class T, int D>
     FixedVector<T,D> operator-(const FixedVector<T,D>& a,const FixedVector<T,D>& b);
template<class T, int D>
     FixedVector<T,D> operator+(const FixedVector<T,D>& a,const FixedVector<T,D>& b);
template<class T, int D>
     FixedVector<T,D> operator*(T a,const FixedVector<T,D>& b);
template<class T, int D>
     FixedVector<T,D> operator*(const FixedVector<T,D>& b,T a);
template<class T, int D>
     FixedVector<T,D> operator/(const FixedVector<T,D>& b,T a);
template<class T, int D>
     T operator*(const FixedVector<T,D>& a,const FixedVector<T,D>& b);
template<class T, int D>
     T norm2(const FixedVector<T,D>& a);
template<class T, int D>
     Real norm(const FixedVector<T,D>& a);
template<class T, int D>
     T max_norm(const FixedVector<T,D>& a);
template<class T, int D>
     T l1_norm(const FixedVector<T,D>& a);


template<class T, int D>
  class FixedVector {
    friend int operator== <> (const FixedVector<T,D>& a, const FixedVector<T,D>& b);
    friend int operator!= <> (const FixedVector<T,D>& a, const FixedVector<T,D>& b);
    friend FixedVector<T,D> operator- <> (const FixedVector<T,D>& a,const FixedVector<T,D>& b);
    friend FixedVector<T,D> operator+ <> (const FixedVector<T,D>& a,const FixedVector<T,D>& b);
    friend FixedVector<T,D> operator* <> (T a,const FixedVector<T,D>& b);
    friend FixedVector<T,D> operator* <> (const FixedVector<T,D>& b,T a);
    friend FixedVector<T,D> operator/ <> (const FixedVector<T,D>& b,T a);
    friend T operator* <> (const FixedVector<T,D>& a,const FixedVector<T,D>& b);
    friend T norm2 <> (const FixedVector<T,D>& a);
    friend Real norm <> (const FixedVector<T,D>& a);
    friend T max_norm <> (const FixedVector<T,D>& a);
    friend T l1_norm <> (const FixedVector<T,D>& a);
  public:
    T x[D];
  public:
/*
    FixedVector(void) {
//    FORALLi {x[i]=(T)0;}
    }
    FixedVector(const FixedVector<T,D>& a) {
      FORALLi {x[i]=a.x[i];}
    }
    void operator=(const FixedVector<T,D>& a) {
      FORALLi {x[i]=a.x[i];}
    }
*/
    void set(T *a) {
      for (int i=0; i<D; i++) {
	x[i]=a[i];
      }
    }
    T * get_buf(void) {
      return x;
    }
    const T& operator()(int i) const {return x[i];}
    T& operator()(int i) {return x[i];}
    const T& operator[](int i) const {return x[i];}
    T& operator[](int i) {return x[i];}
    int get_size(void) {return D;}
    FixedVector<T,D> operator-(void) const {
      RESULT_FORALLi( result.x[i]=-x[i] );
    }
    Real normalize(void) {
      Real norm=0.;
      { FORALLi { norm+=(Real)(x[i]*x[i]); } }
      norm=sqrt(norm);
      { FORALLi {x[i]=(T)((Real)x[i]/norm);} }
      return norm;
    }
    void operator+=(const FixedVector<T,D>& a) {
      FORALLi { x[i]+=a.x[i]; }
    }
    void operator-=(const FixedVector<T,D>& a) {
      FORALLi { x[i]-=a.x[i]; }
    }
  };


 template<class T, int D>
     int operator==(const FixedVector<T,D>& a, const FixedVector<T,D>& b) {
      FORALLi {if (a.x[i]!=b.x[i]) return 0;}
      return 1;
    }

 template<class T, int D>
     int operator!=(const FixedVector<T,D>& a, const FixedVector<T,D>& b) {
      FORALLi {if (a.x[i]!=b.x[i]) return 1;}
      return 0;
    }

 template<class T, int D>
     FixedVector<T,D> operator-(const FixedVector<T,D>& a,const FixedVector<T,D>& b) {
      RESULT_FORALLi( result.x[i]=a.x[i]-b.x[i] );
    }

 template<class T, int D>
     FixedVector<T,D> operator+(const FixedVector<T,D>& a,const FixedVector<T,D>& b) {
      RESULT_FORALLi( result.x[i]=a.x[i]+b.x[i] );
    }

 template<class T, int D>
     FixedVector<T,D> operator*(T a,const FixedVector<T,D>& b) {
      RESULT_FORALLi( result.x[i]=a*b.x[i] );
    }

 template<class T, int D>
     FixedVector<T,D> operator*(const FixedVector<T,D>& b,T a) {
      RESULT_FORALLi( result.x[i]=b.x[i]*a );
    }

 template<class T, int D>
     FixedVector<T,D> operator/(const FixedVector<T,D>& b,T a) {
      RESULT_FORALLi( result.x[i]=b.x[i]/a );
    }

 template<class T, int D>
     T operator*(const FixedVector<T,D>& a,const FixedVector<T,D>& b) {
      T result=(T)0;
      FORALLi { result+=a.x[i]*b.x[i]; }
      return result;
    }

 template<class T, int D>
     T norm2(const FixedVector<T,D>& a) {
      T result=(T)0;
      FORALLi { result+=a.x[i]*a.x[i]; }
      return result;
    }

 template<class T, int D>
     Real norm(const FixedVector<T,D>& a) {
      Real result=0.;
      FORALLi { result+=(Real)(a.x[i]*a.x[i]); }
      return sqrt(result);
    }

 template<class T, int D>
     T max_norm(const FixedVector<T,D>& a) {
      T result=0;
      FORALLi { if (ABS(a.x[i])>result) result=ABS(a.x[i]); }
      return result;
    }

 template<class T, int D>
     T l1_norm(const FixedVector<T,D>& a) {
      T result=0;
      FORALLi { result+=ABS(a.x[i]); }
      return result;
    }


template<class T>
  class Vector2d; 

template<class T>
     T operator^(const FixedVector<T,2>& a,const FixedVector<T,2>& b);


template<class T>
  class Vector2d: public FixedVector<T,2> {
    friend T operator^ <> (const FixedVector<T,2>& a,const FixedVector<T,2>& b);
  public:
    using FixedVector<T,2>::x;
    Vector2d(void) {}
    Vector2d(T _x, T _y) {x[0]=_x; x[1]=_y;}
    Vector2d(const FixedVector<T,2>& a) {
      x[0]=a.x[0];
      x[1]=a.x[1];
    }
    void operator=(const FixedVector<T,2>& a) {
      x[0]=a.x[0];
      x[1]=a.x[1];
    }
    //cross product;
  };


 template<class T>
     T operator^(const FixedVector<T,2>& a,const FixedVector<T,2>& b) {
      return a.x[0]*b.x[1]-a.x[1]*b.x[0];
    }


template<class T>
  class Vector3d; 

template<class T>
     Vector3d<T> operator^(const FixedVector<T,3>& a,const FixedVector<T,3>& b);


template<class T>
  class Vector3d: public FixedVector<T,3> {
    friend Vector3d<T> operator^ <> (const FixedVector<T,3>& a,const FixedVector<T,3>& b);
  public:
    using FixedVector<T,3>::x;
    Vector3d(void) {}
    Vector3d(T _x, T _y, T _z) {x[0]=_x; x[1]=_y; x[2]=_z;}
    Vector3d(const FixedVector<T,3>& a) {
      x[0]=a.x[0];
      x[1]=a.x[1];
      x[2]=a.x[2];
    }
    void operator=(const FixedVector<T,3>& a) {
      x[0]=a.x[0];
      x[1]=a.x[1];
      x[2]=a.x[2];
    }
    //cross product;
  };


 template<class T>
     Vector3d<T> operator^(const FixedVector<T,3>& a,const FixedVector<T,3>& b) {
      return Vector3d<T>(
		      a.x[1]*b.x[2]-a.x[2]*b.x[1],
		      a.x[2]*b.x[0]-a.x[0]*b.x[2],
		      a.x[0]*b.x[1]-a.x[1]*b.x[0]
		      );
    }


#ifdef STREAM_VECTOR

#include <iostream>

template<class T>
istream& operator>>(istream& s, Vector2d<T> &v) {
  s >> v.x[0] >> v.x[1];
  return s;
}

template<class T>
istream& operator>>(istream& s, Vector3d<T> &v) {
  s >> v.x[0] >> v.x[1] >> v.x[2];
  return s;
}

template<class T>
ostream& operator<<(ostream& s, const FixedVector<T,2> &v) {
  s << v.x[0] << " " << v.x[1];
  return s;
}

template<class T>
ostream& operator<<(ostream& s, const FixedVector<T,3> &v) {
  s << v.x[0] << " " << v.x[1] << " " << v.x[2];
  return s;
}

#endif

#define RESULT_FORALLij(body) \
  FixedMatrix<T,D> result;\
  for (int i=0; i<D; i++) {\
    for (int j=0; j<D; j++) {\
      {body;} \
    } \
  }\
  return result

#define FORALLij \
  for(int i=0; i<D; i++)\
    for(int j=0; j<D; j++)

template<class T, int D>
  class FixedMatrix; 

template<class T, int D>
     T max_norm(const FixedMatrix<T,D>& a);
template<class T, int D>
     Real norm(const FixedMatrix<T,D>& a);
template<class T, int D>
     FixedMatrix<T,D> operator-(const FixedMatrix<T,D>& a,const FixedMatrix<T,D>& b);
template<class T, int D>
     FixedMatrix<T,D> operator+(const FixedMatrix<T,D>& a,const FixedMatrix<T,D>& b);
template<class T, int D>
     FixedMatrix<T,D> operator*(T k,const FixedMatrix<T,D>& b);
template<class T, int D>
     FixedMatrix<T,D> operator*(const FixedMatrix<T,D>& b,T k);
template<class T, int D>
     FixedMatrix<T,D> operator/(const FixedMatrix<T,D>& b,T k);
template<class T, int D>
     FixedMatrix<T,D> operator*(const FixedMatrix<T,D>& a,const FixedMatrix<T,D>& b);
template<class T, int D>
     FixedVector<T,D> operator*(const FixedMatrix<T,D>& m, const FixedVector<T,D>& v);
template<class T, int D>
     FixedMatrix<T,D> operator~(const FixedMatrix<T,D>& a);
template<class T, int D>
     T trace(const FixedMatrix<T,D>& a);
template<class T, int D>
     int operator==(const FixedMatrix<T,D>& a, const FixedMatrix<T,D>& b);


template<class T, int D>
  class FixedMatrix {
    friend T max_norm <> (const FixedMatrix<T,D>& a);
    friend Real norm <> (const FixedMatrix<T,D>& a);
    friend FixedMatrix<T,D> operator- <> (const FixedMatrix<T,D>& a,const FixedMatrix<T,D>& b);
    friend FixedMatrix<T,D> operator+ <> (const FixedMatrix<T,D>& a,const FixedMatrix<T,D>& b);
    friend FixedMatrix<T,D> operator* <> (T k,const FixedMatrix<T,D>& b);
    friend FixedMatrix<T,D> operator* <> (const FixedMatrix<T,D>& b,T k);
    friend FixedMatrix<T,D> operator/ <> (const FixedMatrix<T,D>& b,T k);
    friend FixedMatrix<T,D> operator* <> (const FixedMatrix<T,D>& a,const FixedMatrix<T,D>& b);
    friend FixedVector<T,D> operator* <> (const FixedMatrix<T,D>& m, const FixedVector<T,D>& v);
    friend FixedMatrix<T,D> operator~ <> (const FixedMatrix<T,D>& a);
    friend T trace <> (const FixedMatrix<T,D>& a);
    friend int operator== <> (const FixedMatrix<T,D>& a, const FixedMatrix<T,D>& b);
  public:
    T x[D][D];
  public:
/*
    FixedMatrix(void) {}
    FixedMatrix(const FixedMatrix<T,D>& a) {
      FORALLij {x[i][j]=a.x[i][j];}
    }
    void operator=(const FixedMatrix<T,D>& a) {
      FORALLij {x[i][j]=a.x[i][j];}
    }
*/
    void set(T *a) {
      int k=0;
      for (int i=0; i<D; i++) {
	for (int j=0; j<D; j++) {
	  x[i][j]=a[k];
	  k++;
	}
      }
    }
    T& operator()(int i,int j) {return x[i][j];}
    const T& operator()(int i,int j) const {return x[i][j];}
    FixedMatrix<T,D> operator-(void) const {
      RESULT_FORALLij( result.x[i][j]=-x[i][j] );
    }
    void set_column(int column, const FixedVector<T,D> &vector) {
      FORALLi {x[i][column]=vector.x[i];}
    }
    void set_row(int row, const FixedVector<T,D> &vector) {
      FORALLi {x[row][i]=vector.x[i];}
    }
    FixedVector<T,D> get_column(int column) const {
      FixedVector<T,D> result;
      FORALLi {result.x[i]=x[i][column];}
      return result;
    }
    FixedVector<T,D> get_row(int row) const {
      FixedVector<T,D> result;
      FORALLi {result.x[i]=x[row][i];}
      return result;
    }
    void diag(const FixedVector<T,D> &v) {
      zero();
      FORALLi {(*this)(i,i)=v(i);}
    }
    //transpose;
    void identity(void) {
      FORALLij {x[i][j]=(i==j ? 1 : 0);}
    }
    void zero(void) {
      FORALLij {x[i][j]=(T)0.;}
    }
  };


 template<class T, int D>
     T max_norm(const FixedMatrix<T,D>& a) {
      T result=0;
      FORALLij { if (ABS(a.x[i][j])>result) result=ABS(a.x[i][j]); }
      return result;
    }

 template<class T, int D>
     Real norm(const FixedMatrix<T,D>& a) {
      Real result=0;
      FORALLij { result+=(Real)(a.x[i][j]*a.x[i][j]); }
      return sqrt(result);
    }

 template<class T, int D>
     FixedMatrix<T,D> operator-(const FixedMatrix<T,D>& a,const FixedMatrix<T,D>& b) {
      RESULT_FORALLij( result.x[i][j]=a.x[i][j]-b.x[i][j] );
    }

 template<class T, int D>
     FixedMatrix<T,D> operator+(const FixedMatrix<T,D>& a,const FixedMatrix<T,D>& b) {
      RESULT_FORALLij( result.x[i][j]=a.x[i][j]+b.x[i][j] );
    }

 template<class T, int D>
     FixedMatrix<T,D> operator*(T k,const FixedMatrix<T,D>& b) {
      RESULT_FORALLij( result.x[i][j]=k*b.x[i][j] );
    }

 template<class T, int D>
     FixedMatrix<T,D> operator*(const FixedMatrix<T,D>& b,T k) {
      RESULT_FORALLij( result.x[i][j]=b.x[i][j]*k );
    }

 template<class T, int D>
     FixedMatrix<T,D> operator/(const FixedMatrix<T,D>& b,T k) {
      RESULT_FORALLij( result.x[i][j]=b.x[i][j]/k );
    }

 template<class T, int D>
     FixedMatrix<T,D> operator*(const FixedMatrix<T,D>& a,const FixedMatrix<T,D>& b) {
      RESULT_FORALLij(
		      result.x[i][j]=(T)0;
		      for (int k=0; k<D; k++) {
		      result.x[i][j]+=a.x[i][k]*b.x[k][j];
		      }
		      );
    }

 template<class T, int D>
     FixedVector<T,D> operator*(const FixedMatrix<T,D>& m, const FixedVector<T,D>& v) {
      FixedVector<T,D> result;
      for (int i=0; i<D; i++) {
	result.x[i]=(T)0;
	for (int j=0; j<D; j++) {
	  result.x[i]+=m.x[i][j]*v.x[j];
	}
      }
      return result;
    }

 template<class T, int D>
     FixedMatrix<T,D> operator~(const FixedMatrix<T,D>& a) {
      RESULT_FORALLij( result.x[i][j]=a.x[j][i] );
    }

 template<class T, int D>
     T trace(const FixedMatrix<T,D>& a) {
      T sum=(T)0;
      for (int i=0; i<D; i++) {
	sum+=a.x[i][i];
      }
      return sum;
    }

 template<class T, int D>
     int operator==(const FixedMatrix<T,D>& a, const FixedMatrix<T,D>& b) {
      FORALLij {if (a.x[i][j]!=b.x[i][j]) return 0;}
      return 1;
    }


template<class T>
  class Matrix2d; 

template<class T>
     T det(const FixedMatrix<T,2>& a);
template<class T>
     Matrix2d<T> operator!(const FixedMatrix<T,2>& a);


template<class T>
  class Matrix2d: public FixedMatrix<T,2> {
    friend T det <> (const FixedMatrix<T,2>& a);
    friend Matrix2d<T> operator! <> (const FixedMatrix<T,2>& a);
  public:
    using FixedMatrix<T,2>::x;
    Matrix2d(void) {}
    Matrix2d(const FixedMatrix<T,2>& a) {
      x[0][0]=a.x[0][0];
      x[0][1]=a.x[0][1];
      x[1][0]=a.x[1][0];
      x[1][1]=a.x[1][1];
    }
    Matrix2d(T a, T b, T c, T d) {
      x[0][0]=a;
      x[0][1]=b;
      x[1][0]=c;
      x[1][1]=d;
    }
    void set_diagonal(T a, T b) {
      FixedMatrix<T,2>::zero();
      x[0][0]=a;
      x[1][1]=b;
    }
    void operator=(const FixedMatrix<T,2>& a) {
      x[0][0]=a.x[0][0];
      x[0][1]=a.x[0][1];
      x[1][0]=a.x[1][0];
      x[1][1]=a.x[1][1];
    }
    //inverse;
  };


 template<class T>
     T det(const FixedMatrix<T,2>& a) {
      return a.x[0][0]*a.x[1][1]-a.x[1][0]*a.x[0][1];
    }

 template<class T>
     Matrix2d<T> operator!(const FixedMatrix<T,2>& a) {
      T d=det(a);
      Matrix2d<T> result;
      result.x[0][0]= a.x[1][1]/d;
      result.x[0][1]=-a.x[0][1]/d;
      result.x[1][0]=-a.x[1][0]/d;
      result.x[1][1]= a.x[0][0]/d;
      return result;
    }


template<class T>
  class Matrix3d; 

template<class T>
     T det(const FixedMatrix<T,3>& a);
template<class T>
     FixedMatrix<T,3> operator!(const FixedMatrix<T,3>& a);


template<class T>
  class Matrix3d: public FixedMatrix<T,3> {
    friend T det <> (const FixedMatrix<T,3>& a);
    friend FixedMatrix<T,3> operator! <> (const FixedMatrix<T,3>& a);
  public:
    using FixedMatrix<T,3>::x;
    Matrix3d(void) {}
    Matrix3d(const FixedMatrix<T,3>& a) {
      for (int i=0; i<3; i++) {
	for (int j=0; j<3; j++) {
	  x[i][j]=a.x[i][j];
	}
      }
    }
    void set_diagonal(T a, T b, T c) {
      FixedMatrix<T,3>::zero();
      x[0][0]=a;
      x[1][1]=b;
      x[2][2]=c;
    }
    void operator=(const FixedMatrix<T,3>& a) {
      for (int i=0; i<3; i++) {
	for (int j=0; j<3; j++) {
	  x[i][j]=a.x[i][j];
	}
      }
    }
    //inverse;
  };


 template<class T>
     T det(const FixedMatrix<T,3>& a) {
      return
	a.x[0][0] * (a.x[1][1]*a.x[2][2] -a.x[1][2]*a.x[2][1])+
	a.x[0][1] * (a.x[1][2]*a.x[2][0] -a.x[1][0]*a.x[2][2])+
	a.x[0][2] * (a.x[1][0]*a.x[2][1] -a.x[1][1]*a.x[2][0]);
    }

 template<class T>
     FixedMatrix<T,3> operator!(const FixedMatrix<T,3>& a) {
      T d=det(a);
      Matrix3d<T> result;
      result.x[0][0]=(+a.x[1][1]*a.x[2][2]-a.x[1][2]*a.x[2][1])/d;
      result.x[0][1]=(-a.x[0][1]*a.x[2][2]+a.x[0][2]*a.x[2][1])/d;
      result.x[0][2]=(+a.x[0][1]*a.x[1][2]-a.x[0][2]*a.x[1][1])/d;
      result.x[1][0]=(-a.x[1][0]*a.x[2][2]+a.x[1][2]*a.x[2][0])/d;
      result.x[1][1]=(+a.x[0][0]*a.x[2][2]-a.x[0][2]*a.x[2][0])/d;
      result.x[1][2]=(-a.x[0][0]*a.x[1][2]+a.x[0][2]*a.x[1][0])/d;
      result.x[2][0]=(+a.x[1][0]*a.x[2][1]-a.x[1][1]*a.x[2][0])/d;
      result.x[2][1]=(-a.x[0][0]*a.x[2][1]+a.x[0][1]*a.x[2][0])/d;
      result.x[2][2]=(+a.x[0][0]*a.x[1][1]-a.x[0][1]*a.x[1][0])/d;
      return result;
    }


#ifdef STREAM_VECTOR

template<class T>
ostream& operator<<(ostream& s, const FixedMatrix<T,2> &a) {
  for (int i=0; i<2; i++) {
    for (int j=0; j<2; j++) {
      s << a.x[i][j] << " ";
    }
    s << endl;
  }
  return s;
}

template<class T>
ostream& operator<<(ostream& s, const FixedMatrix<T,3> &a) {
  for (int i=0; i<3; i++) {
    for (int j=0; j<3; j++) {
      s << a.x[i][j] << " ";
    }
    s << endl;
  }
  return s;
}

template<class T>
istream& operator>>(istream& s, Matrix2d<T> &a) {
  for (int i=0; i<2; i++) {
    for (int j=0; j<2; j++) {
      s >> a.x[i][j];
    }
  }
  return s;
}

template<class T>
istream& operator>>(istream& s, Matrix3d<T> &a) {
  for (int i=0; i<3; i++) {
    for (int j=0; j<3; j++) {
      s >> a.x[i][j];
    }
  }
  return s;
}

#endif

template<class T, int D>
  class BoundingBox {
  public:
    FixedVector<T,D> min,max;
  public:
    BoundingBox(void) {}
    BoundingBox(const FixedVector<T,D> &_min,const FixedVector<T,D> &_max): min(_min), max(_max) {}
    inline void before_update(void);
    void update(const FixedVector<T,D>& v) {
      FORALLi {
	if (v.x[i]<min.x[i]) min.x[i]=v.x[i];
	if (v.x[i]>max.x[i]) max.x[i]=v.x[i];
      }
    }
    int is_in(const FixedVector<T,D> &v) const {
      FORALLi {
	if (v.x[i]<min.x[i]) return 0;
	if (v.x[i]>max.x[i]) return 0;
      }
      return 1;
    }
  };

template<> inline void BoundingBox<int,3>::before_update(void) {
  for (int i=0; i<3; i++) {
    min.x[i]= MAXINT;
    max.x[i]=-MAXINT;
  }
}

template<> inline void BoundingBox<Real,3>::before_update(void) {
  for (int i=0; i<3; i++) {
    min.x[i]= MAXFLOAT;
    max.x[i]=-MAXFLOAT;
  }
}

template<> inline void BoundingBox<int,2>::before_update(void) {
  for (int i=0; i<2; i++) {
    min.x[i]= MAXINT;
    max.x[i]=-MAXINT;
  }
}

template<> inline void BoundingBox<Real,2>::before_update(void) {
  for (int i=0; i<2; i++) {
    min.x[i]= MAXFLOAT;
    max.x[i]=-MAXFLOAT;
  }
}

#endif
