 function res=plotmag(n)
 vv=load('volume.0');  
 
 yy=[];
 for L=1:n
  ff=['mag.', num2str(L)]; 
  asd=load(ff);
  yy=[yy, asd];
 end

 figure, plot(vv, yy, '-o'), title(' magnetic moment for each atoms ') 
 saveas(gcf, 'fig33_V_mag.png') 
 res=[vv, yy];

