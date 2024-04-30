%  EOS fittings by Shunli Shang at PSU, 06-Aug-2008
%  1:   4-parameter (Teter-Shang) mBM4  1 
%  2:   5-parameter (Teter-Shang) mBM5  2
%   3: 4-parameter                BM4   3 
%   4: 5-parameter                BM5   4 
%  5: 4-parameter Natural         Log4  5
%  6: 5-parameter Natural         Log5  6
%  7:  4-parameter Murnaghan      Mur   7
%  8:  4-parameter Vinet          Vinet 8 
%  9:  4-parameter Morse          Morse 9 
% get_0kres.m, includes function: eosfit.m
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
%  
 ipress =  1; % > 0, including the pressure data to compare
 idata  =  2; % 1: E-V data in one file, 2: E-V data in 2 file
 istrain=  1; % > 0 plot the strain vs. volume relation
 LL0    =  1; % the no. of data point used as the reference state
%%%
if idata==1
    data=load('ev.0'); volume=data(:,1); fzero=data(:,2);
 end
 if idata ==2 
    fzero   =load('energy.0');    % eV/UC
    volume  =load('volume.0');    % A^3  
    volume123=volume; 
 end
 if ipress > 0; 
     press0  =load('pressure.0');  % kBar
     pressure=press0/10;           % GPa
 end
 if ipress < 0; pressure=zeros(size(volume)); end
 
 nn1=1;     %%%%%%%%%%% <<<<<<<<<<<<<<<<<<<<<<<
 nn2=999;     %%%%%%%%%%% <<<<<<<<<<<<<<<<<<<<<<<
% if exist('~/asdf/eosrange.1');
%  aaa=load('~/asdf/eosrange.1'); nn1=aaa(1); nn2=aaa(2);
% end
 
 nn3=length(volume), if nn3 < nn2; nn2=nn3; end
 datarange=nn1:nn2;
 
 iselect = 1;     %%%%%%%%%% <<<<<<<<<<<<<<<<<<<<<<<
  if iselect > 0;
     datarange =[    4 5 6 7 8 9 10 ],   %%%%%  <<<<<<<<<
  end
   
 fzero=fzero(datarange); volume=volume(datarange); pressure=pressure(datarange);
 
 ifigure = -9;      % > 0 plot figure
 isave   = -9;      % > 0 save volume and fitted energy, pressure 
 vv=(min(volume)-1):0.05:(max(volume)+2); % volume for the final-fited E-V
 numbereos=9;        % 6 ( linear fittings);  9: all fittings
if numbereos==6;  ieos=[1 2 3 4 5 6];  end
if numbereos==9;  ieos=[1 2 3 4 5 6 7 8 9]; end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
if istrain > 0
 scales =load('scales.0');
 vectors=load('vectors.0');
 sss=[];
   L=LL0;
   mm0=3*(L-1)+1; mm1=3*L;
   asd0=scales(L)*vectors(mm0:mm1,:);
 for L=1:length(scales);
   mm0=3*(L-1)+1; mm1=3*L;
   aa=scales(L)*vectors(mm0:mm1,:);
   zz=inv(asd0)*aa; zz=reshape(zz, 1,9);  
   sss=[sss; zz];
 end
 figure, plot(volume123, sss, '-o'); title('strain relation vs volume'); 
 saveas(gcf, 'fig1_strain_vol.png')
end; % end of 'if istrain > 0'

%%%%%%%%%%%%%%%%%
 chunit=160.2189;
 vzero=mean(volume); 
 res=[];  resdiff=[]; resee=[]; resxx=[]; resdiffpp=[]; respp=[];
%

for L = ieos 
%%%%%% 
if L==1   %%%   mBM4 
   L;
 bb=fzero;
 AA=[ones(size(volume)), volume.^(-1/3), volume.^(-2/3), volume.^(-1)];  %(nx4)  
 xx=pinv(AA)*bb;   %(4x1)=(4xn)*(nx1), solve it by pseudo-inversion: Ax=b
 a=xx(1);  b=xx(2);  c=xx(3);  d=xx(4); e=0.0;  xx=[a b c d e];

 pp=(4*e)./(3*vv.^(7/3)) + d./vv.^2 + (2*c)./(3*vv.^(5/3)) + b./(3*vv.^(4/3));  pp=pp*chunit; 
 pp0= (4*e)./(3*volume.^(7/3)) + d./volume.^2 + (2*c)./(3*volume.^(5/3)) + b./(3*volume.^(4/3)); 
 pp0=pp0*chunit; 
 if ifigure >1; figure, plot(vv,pp, volume, pressure, 'o'), title('P-V curve, mBM4, No. 1, GPa'),
 saveas(gcf, 'figB_mBM4_PV.png')
 end
 diffp=pp0-pressure;
 resdiffpp=[resdiffpp, diffp];
 
 ee = a + b*(vv).^(-1/3) + c*(vv).^(-2/3) + d*(vv).^(-1) + e*(vv).^(-4/3); 
 if ifigure > 1; figure, plot(vv,ee, volume, bb,'o'), title('E-V curve, mBM4, No. 1'), 
 saveas(gcf, 'figC_mBM4_EV.png')
 end;
 curve= a + b*(volume).^(-1/3) + c*(volume).^(-2/3) + d*(volume).^(-1) + e*(volume).^(-4/3);
 diff = curve-bb;
 prop = eosparameter45(xx, vzero, L); % [V0, E0, P, B, BP, B2P];
 vzero=prop(1);
 
 newxx   =build_eos(prop, L);
 resxx   =[resxx; L, xx; L, newxx];
 
 nnn=length(bb);
 qwe=(diff./bb).^2; 
 asd=sqrt(sum(qwe/nnn)); 
 res     =[res; L, prop, asd];
 
 resee   =[resee; ee]; respp=[respp; pp];
 resdiff =[resdiff, diff]; 
 
 xini = [prop(1), prop(2), prop(4)/chunit, prop(5)]; % used for ieos= 7 and 8
                          %   V=xout(1); E0=xout(2); B=xout(4); bp=xout(5); 
end % end of L==1 
 
%%%%%% 
if L == 2  %% mBM5 
   L;
 bb=fzero;
 AA=[ones(size(volume)), volume.^(-1/3), volume.^(-2/3), volume.^(-1), volume.^(-4/3)];  %(nx5)  
 xx=pinv(AA)*bb;   %(4x1)=(4xn)*(nx1), solve it by pseudo-inversion: Ax=b
 a=xx(1);  b=xx(2);  c=xx(3);  d=xx(4); e=xx(5);  xx=[a b c d e];
 
 pp=(4*e)./(3*vv.^(7/3)) + d./vv.^2 + (2*c)./(3*vv.^(5/3)) + b./(3*vv.^(4/3));  pp=pp*chunit; 
 pp0= (4*e)./(3*volume.^(7/3)) + d./volume.^2 + (2*c)./(3*volume.^(5/3)) + b./(3*volume.^(4/3)); 
 pp0=pp0*chunit; 
 if ifigure >1; figure, plot(vv,pp, volume, pressure, 'o'), title('P-V curve, mBM5, No. 2, GPa'), 
 saveas(gcf, 'figD_mBM5_PV.png')
 end
 diffp=pp0-pressure;
 resdiffpp=[resdiffpp, diffp];
 
 ee = a + b*(vv).^(-1/3) + c*(vv).^(-2/3) + d*(vv).^(-1) + e*(vv).^(-4/3); 
 if ifigure > 1; figure, plot(vv,ee, volume, bb,'o'), title('E-V curve, mBM5, No. 2'), 
 saveas(gcf, 'figE_mBM5_EV.png')
 end;
 curve= a + b*(volume).^(-1/3) + c*(volume).^(-2/3) + d*(volume).^(-1) + e*(volume).^(-4/3);
 diff = curve-bb;
 prop =eosparameter45(xx, vzero, L); % [V0, E0, P, B, BP, B2P];
 
 newxx   =build_eos(prop, L);
 resxx   =[resxx; L, xx; L, newxx];
 
 
 nnn=length(bb);
 qwe=(diff./bb).^2; 
 asd=sqrt(sum(qwe/nnn)); 
 res     =[res; L, prop, asd];
 resee   =[resee; ee]; respp=[respp; pp];
 resdiff =[resdiff, diff]; 
end % end of L==2 

%%%%%% 
if L == 3  %% BM4 
   L;
 bb=fzero;
 AA=[ones(size(volume)), volume.^(-2/3), volume.^(-4/3), volume.^(-2)];  %(nx4)  
 xx=pinv(AA)*bb;   %(4x1)=(4xn)*(nx1), solve it by pseudo-inversion: Ax=b
 a=xx(1);  b=xx(2);  c=xx(3);  d=xx(4); e=0.0;  xx=[a b c d e];
 
 x=vv;
 pp=((8*e)./(3*x.^(11/3))+(2*d)./x.^3+(4*c)./(3*x.^(7/3))+(2*b)./(3*x.^(5/3)))*chunit;
 x=volume;
 pp0=((8*e)./(3*x.^(11/3))+(2*d)./x.^3+(4*c)./(3*x.^(7/3))+(2*b)./(3*x.^(5/3)))*chunit;
 if ifigure >1; figure, plot(vv,pp, volume, pressure, 'o'), title('P-V curve, BM4, No. 3, GPa'), end
 diffp=pp0-pressure;
 resdiffpp=[resdiffpp, diffp];
 
 ee = a + b*(vv).^(-2/3) + c*(vv).^(-4/3) + d*(vv).^(-2) + e*(vv).^(-8/3);
 if ifigure > 1; figure, plot(vv,ee, volume, bb,'o'), title('E-V curve, BM4, No. 3'), end;
 curve= a + b*(volume).^(-2/3) + c*(volume).^(-4/3) + d*(volume).^(-2) + e*(volume).^(-8/3);
 diff = curve-bb;
 prop= eosparameter45(xx, vzero, L); % [V0, E0, P, B, BP, B2P]; 
 
 bm4_v0=prop(1); bm4_B0=prop(4); bm4_Bp=prop(5);

 newxx   =build_eos(prop, L);
 resxx   =[resxx; L, xx; L, newxx];
 
 nnn=length(bb);
 qwe=(diff./bb).^2; 
 asd=sqrt(sum(qwe/nnn)); 
 res     =[res; L, prop, asd];
 resee   =[resee; ee]; respp=[respp; pp];
 resdiff =[resdiff, diff]; 
end % end of L==3 

%%%%%% 
if L == 4  %% BM5
   L;
 bb=fzero;
 AA=[ones(size(volume)), volume.^(-2/3), volume.^(-4/3), volume.^(-2), volume.^(-8/3)];  %(nx4)  
 xx=pinv(AA)*bb;   %(4x1)=(4xn)*(nx1), solve it by pseudo-inversion: Ax=b
 a=xx(1);  b=xx(2);  c=xx(3);  d=xx(4); e=xx(5);  xx=[a b c d e];
 
 x=vv;
 pp=((8*e)./(3*x.^(11/3))+(2*d)./x.^3+(4*c)./(3*x.^(7/3))+(2*b)./(3*x.^(5/3)))*chunit;
 x=volume;
 pp0=((8*e)./(3*x.^(11/3))+(2*d)./x.^3+(4*c)./(3*x.^(7/3))+(2*b)./(3*x.^(5/3)))*chunit;
 if ifigure >1; figure, plot(vv,pp, volume, pressure, 'o'), title('P-V curve, BM5, No. 4, GPa'), end
 diffp=pp0-pressure;
 resdiffpp=[resdiffpp, diffp];
 
 ee = a + b*(vv).^(-2/3) + c*(vv).^(-4/3) + d*(vv).^(-2) + e*(vv).^(-8/3); 
 if ifigure > 1; figure, plot(vv,ee, volume, bb,'o'), title('E-V curve, BM5, No. 4'), end;
 curve= a + b*(volume).^(-2/3) + c*(volume).^(-4/3) + d*(volume).^(-2) + e*(volume).^(-8/3);
 diff = curve-bb;
 prop=eosparameter45(xx, vzero, L); % [V0, E0, P, B, BP, B2P];
 
 newxx   =build_eos(prop, L);
 resxx   =[resxx; L, xx; L, newxx];
 
 nnn=length(bb);
 qwe=(diff./bb).^2; 
 asd=sqrt(sum(qwe/nnn)); 
 res     =[res; L, prop, asd];
 resee   =[resee; ee];  respp=[respp; pp];
 resdiff =[resdiff, diff]; 
end % end of L==4 

%%%%%% 
if L == 5  %% LOG4
   L;
 bb=fzero;
 AA=[ones(size(volume)), log(volume), log(volume).^2, log(volume).^3];  %(nx4)  
 xx=pinv(AA)*bb;   %(4x1)=(4xn)*(nx1), solve it by pseudo-inversion: Ax=b
 a=xx(1);  b=xx(2);  c=xx(3);  d=xx(4); e=0.0;  xx=[a b c d e];
 
 x=vv;
 pp=(-((b + 2*c*log(x) + 3*d*log(x).^2 + 4*e*log(x).^3)./x))*chunit;
 x=volume;
 pp0=(-((b + 2*c*log(x) + 3*d*log(x).^2 + 4*e*log(x).^3)./x))*chunit;
 if ifigure >1; figure, plot(vv,pp, volume, pressure, 'o'), title('P-V curve, LOG4, No. 5, GPa'), end
 diffp=pp0-pressure;
 resdiffpp=[resdiffpp, diffp];
 
 ee = a + b*log(vv) + c*log(vv).^2 + d*log(vv).^3 + e*log(vv).^4;
 if ifigure > 1; figure, plot(vv,ee, volume, bb,'o'), title('E-V curve, LOG4, No. 5'), end;
 curve= a + b*log(volume) + c*log(volume).^2 + d*log(volume).^3 + e*log(volume).^4;
 diff = curve-bb;
 prop=eosparameter45(xx, vzero, L); % [V0, E0, P, B, BP, B2P]; 
 
 newxx   =build_eos(prop, L);
 resxx   =[resxx; L, xx; L, newxx];
 
 nnn=length(bb);
 qwe=(diff./bb).^2; 
 asd=sqrt(sum(qwe/nnn)); 
 res     =[res; L, prop, asd];
 resee   =[resee; ee];  respp=[respp; pp];
 resdiff =[resdiff, diff]; 
end % end of L==5 
 
%%%%%% 
if L == 6  %% LOG5
   L;
 bb=fzero;
 AA=[ones(size(volume)), log(volume), log(volume).^2, log(volume).^3, log(volume).^4];  %(nx4)  
 xx=pinv(AA)*bb;   %(4x1)=(4xn)*(nx1), solve it by pseudo-inversion: Ax=b
 a=xx(1);  b=xx(2);  c=xx(3);  d=xx(4); e=xx(5);  xx=[a b c d e];
 
 x=vv;
 pp=(-((b + 2*c*log(x) + 3*d*log(x).^2 + 4*e*log(x).^3)./x))*chunit;
 x=volume;
 pp0=(-((b + 2*c*log(x) + 3*d*log(x).^2 + 4*e*log(x).^3)./x))*chunit;
 if ifigure >1; figure, plot(vv,pp, volume, pressure, 'o'), title('P-V curve, LOG5, No. 6, GPa'), end
 diffp=pp0-pressure;
 resdiffpp=[resdiffpp, diffp];
 
 ee = a + b*log(vv) + c*log(vv).^2 + d*log(vv).^3 + e*log(vv).^4; 
 if ifigure > 1; figure, plot(vv,ee, volume, bb,'o'), title('E-V curve, LOG5, No. 6'), end;
 curve= a + b*log(volume) + c*log(volume).^2 + d*log(volume).^3 + e*log(volume).^4;
 diff = curve-bb;
 prop=eosparameter45(xx, vzero, L); % [V0, E0, P, B, BP, B2P]; 
  
 newxx   =build_eos(prop, L);
 resxx   =[resxx; L, xx; L, newxx];
 
 nnn=length(bb);
 qwe=(diff./bb).^2; 
 asd=sqrt(sum(qwe/nnn)); 
 res     =[res; L, prop, asd];
 resee   =[resee; ee];  respp=[respp; pp];
 resdiff =[resdiff, diff]; 
end % end of L==6 
 
%%%%%% 
if L == 7  %% Mur
   L;
  OPT=optimset; OPT.MaxFunEvals=99000; OPT.MaxIter=33000; OPT.TolFun=1e-9; OPT.TolX=1e-9;
  %OPT.LargeScale='off';
   xdata=volume;
   ydata=fzero;
   Data=[xdata, ydata];
   
           % V-1      E-2       B-3           bp-4
   %xini = [prop(1), prop(2), prop(3)/chunit, prop(4)]; % input as ieos==0
   [xout,resnorm] = lsqnonlin(@murnaghan, xini, [], [], OPT, Data);
   %[xout,resnorm] = lsqnonlin(murnaghan,xini,Data);
   V=xout(1); E0=xout(2); B=xout(3); bp=xout(4); 
 
   x=vv;
   pp=chunit*(B*(-1 + (V./x).^bp))./bp;
   x=volume;
   pp0=chunit*(B*(-1 + (V./x).^bp))./bp;
   if ifigure >1; figure, plot(vv,pp, volume, pressure, 'o'), title('P-V curve, Mur, No. 7, GPa'), end
   diffp=pp0-pressure;
   resdiffpp=[resdiffpp, diffp];
   
   ee = (E0-(B*V)/(-1+bp) + (B*(1+(V./vv).^bp./(-1+bp)).*vv)./bp); 
    if ifigure > 1; figure, plot(vv,ee, volume, bb,'o'), title('E-V curve, Mur, No. 7'), end;
   curve= (E0-(B*V)/(-1+bp) + (B*(1+(V./volume).^bp./(-1+bp)).*volume)./bp);
   diff = curve-bb;
 
   prop=[V, E0, 0, B*chunit, bp, 0]; 

   nnn=length(bb);
   qwe=(diff./bb).^2; 
   asd=sqrt(sum(qwe/nnn)); 
   res     =[res; L, prop, asd];  % [V0, E0, P, B, BP, B2P];
   resee   =[resee; ee];  respp=[respp; pp];
   resdiff =[resdiff, diff]; 
end % end of L==7 

%%%%%% 
if L == 8  %% Vinet
   L;
  OPT=optimset; OPT.MaxFunEvals=99000; OPT.MaxIter=33000; OPT.TolFun=1e-9; OPT.TolX=1e-9;
  %OPT.LargeScale='off';
   xdata=volume;
   ydata=fzero;
   Data=[xdata, ydata];
   
           % V-1      E-2       B-3           bp-4
 % xini = [prop(1), prop(2), prop(3)/chunit, prop(4)];  % input at ieos==0
   [xout,resnorm] = lsqnonlin(@vineteos, xini, [], [], OPT, Data);
   V=xout(1); E0=xout(2); B=xout(3); bp=xout(4); 
    
 x=vv;
 pp=chunit*(-3*B*(-1 + (x./V).^(1/3)))./(exp((3*(-1 + bp)*(-1 + (x/V).^(1/3)))/2).*(x/V).^(2/3));
 x=volume;
 pp0=chunit*(-3*B*(-1 + (x./V).^(1/3)))./(exp((3*(-1 + bp)*(-1 + (x/V).^(1/3)))/2).*(x/V).^(2/3));
 if ifigure >1; figure, plot(vv,pp, volume, pressure, 'o'), title('P-V curve, Vinet, No. 8, GPa'), end
 diffp=pp0-pressure;
 resdiffpp=[resdiffpp, diffp];
   
  ee=E0 + (4*B*V)./(-1 + bp).^2 - (4*B*V*(1 + (3*(-1 + bp)*(-1 + ...
    (vv/V).^(1/3)))/2))./((-1 + bp).^2.*exp((3*(-1 + bp).*(-1 + (vv./V).^(1/3)))/2));
   if ifigure > 1; figure, plot(vv,ee, volume, bb,'o'), title('E-V curve, Vinet, No. 8'), end;
 curve=E0 + (4*B*V)./(-1 + bp).^2 - (4*B*V*(1 + (3*(-1 + bp)*(-1 + ...
    (volume/V).^(1/3)))/2))./((-1 + bp).^2.*exp((3*(-1 + bp).*(-1 + (volume./V).^(1/3)))/2));
 
   diff = curve-bb;
  
   b2p= (19 - 18*bp - 9*bp^2)/(36*B);
   prop=[V, E0, 0, B*chunit, bp, b2p/chunit]; 
   
   
   nnn=length(bb);
   qwe=(diff./bb).^2; 
   asd=sqrt(sum(qwe/nnn)); 
   res     =[res; L, prop, asd]; % [V0, E0, P, B, BP, B2P];
   resee   =[resee; ee];  respp=[respp; pp];
   resdiff =[resdiff, diff]; 
end % end of L==8 

%%%%%% 
if L == 9  %% Morse
   L;
  OPT=optimset; OPT.MaxFunEvals=99000; OPT.MaxIter=33000; OPT.TolFun=1e-9; OPT.TolX=1e-9;
  %OPT.LargeScale='off';
   xdata=volume;
   ydata=fzero;
   Data=[xdata, ydata];
   
           % V-1      E-2       B-3           bp-4
 % xini = [prop(1), prop(2), prop(4)/chunit, prop(5)];  % input at ieos==1
   [xout,resnorm] = lsqnonlin(@morseeos,xini,[],[],OPT,Data);
   V=xout(1); E0=xout(2); B=xout(3); bp=xout(4); 
   
   a= E0 + (9*B*V)/(2*(-1 + bp)^2);
   b= (-9*B*exp(-1 + bp)*V)/(-1 + bp)^2;
   c= (9*B*exp(-2 + 2*bp)*V)/(2*(-1 + bp)^2);
   d= (1 - bp)/V^(1/3);
   
 x=vv;
 pp=-chunit*(d.*exp(d.*x.^(1/3)).*(b + 2*c.*exp(d.*x.^(1/3))))./(3.*x.^(2/3)); 
 x=volume;
 pp0=-chunit*(d.*exp(d.*x.^(1/3)).*(b + 2*c.*exp(d.*x.^(1/3))))./(3.*x.^(2/3)); 
 if ifigure >1; figure, plot(vv,pp, volume, pressure, 'o'), title('P-V curve, Morse, No. 9, GPa'), end
 diffp=pp0-pressure;
 resdiffpp=[resdiffpp, diffp];
   
 ee=a + b.*exp(d.*vv.^(1/3)) + c.*exp(2*d.*vv.^(1/3));
  if ifigure > 1; figure, plot(vv,ee, volume, bb,'o'), title('E-V curve, Vinet, No. 8'), end;
  x=volume;
 curve=a + b.*exp(d.*x.^(1/3)) + c.*exp(2*d.*x.^(1/3)); 
   diff = curve-bb;
  
   b2p= (5 - 5*bp - 2*bp^2)/(9*B);
   
   prop=[V, E0, 0, B*chunit, bp, b2p/chunit]; 

    nnn=length(bb);
    qwe=(diff./bb).^2; 
    asd=sqrt(sum(qwe/nnn)); 
    res     =[res; L, prop, asd]; % [V0, E0, P, B, BP, B2P];
   resee   =[resee; ee];  respp=[respp; pp];
   resdiff =[resdiff, diff]; 
end % end of L==9 

%%%%%% 
end % end of for L=ieos

if numbereos==6;
pp4=[respp(1,:); respp(3,:); respp(5,:)]; 
pp5=[respp(2,:); respp(4,:); respp(6,:)]; 
figure, plot(vv, pp4, vv, pp5, '--', volume, pressure,'o'), title('P-V curves'), 
legend('1-mBM4', '3-BM4', '5-LOG4', '2-mBM5','4-BM5','6-LOG5', '10-Calc');
saveas(gcf, 'fig62_PV_fromEV.png')
ee4=[resee(1,:); resee(3,:); resee(5,:)]; 
ee5=[resee(2,:); resee(4,:); resee(6,:)]; 
figure, plot(vv, ee4, vv, ee5, '--', volume, bb,'o'), title('E-V curves')
legend('1-mBM4', '3-BM4', '5-LOG4', '2-mBM5','4-BM5','6-LOG5', '10-Calc');
saveas(gcf, 'fig61_EV.png')
end

if numbereos==9;
pp4=[respp(1,:); respp(3,:); respp(5,:); respp(7,:); respp(8,:); respp(9,:)]; 
pp5=[respp(2,:); respp(4,:); respp(6,:)]; 
figure, plot(vv, pp4, vv, pp5, '--', volume, pressure,'o'), title('P-V curves'), 
legend('1-mBM4', '3-BM4', '5-LOG4', '7-Mur', '8-Vinet', '9-Morse', '2-mBM5','4-BM5','6-LOG5', '10-Calc');
saveas(gcf, 'fig92_PV_fromEV.png')

ee4=[resee(1,:); resee(3,:); resee(5,:); resee(7,:); resee(8,:); resee(9,:)]; 
ee5=[resee(2,:); resee(4,:); resee(6,:)]; 
figure, plot(vv, ee4, vv, ee5, '--', volume, bb,'o'), 
textEV9 = ['E-V curves: BM4 V-B-Bp = ', string(bm4_v0), ' ', string(bm4_B0), ' ', string(bm4_Bp)]; 
% bm4_v0=prop(1); bm4_B0=prop(4); bm4_Bp=prop(5);
title(join(textEV9))
legend('1-mBM4', '3-BM4', '5-LOG4', '7-Mur', '8-Vinet', '9-Morse', '2-mBM5','4-BM5','6-LOG5', '10-Calc');
saveas(gcf, 'fig91_EV.png')
end

resxx;

max_diff= max(abs(resdiffpp));
av_diff =mean(abs(resdiffpp));
dpp_av_max=[av_diff; max_diff]

max_diff= max(abs(resdiff));
av_diff =mean(abs(resdiff));
dene_av_max=[av_diff; max_diff]

qwe=res(:,end)*10^4; res(:,end)=qwe;
fitted_res=[res(:,1:3), res(:,5:end)];

%ff=['out_eosres'];   eval(['save ' ff ' fitted_res -ascii']); 

diff_av_max=[res(:,1), dene_av_max', dpp_av_max']; 
%ff=['out_eosdiff'];  eval(['save ' ff ' diff_av_max -ascii']); 

if isave > 0;  
 resve =[vv', resee'];
 resvp =[vv', respp']; 
 ff=['out_fit_VE'];  eval(['save ' ff ' resve -ascii']); 
 ff=['out_fit_VP'];  eval(['save ' ff ' resvp -ascii']); 
end

%
%  N-list  V0  E0   B  BP  B2P  av_diff_e  max_diff_e  av_diff_p   max_diff_p   
%    1     2   3    4  5   6        2        3          4          5                     

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if ipress > 0 
pvfit_res=pveosfit(volume, fzero, pressure, vv, isave, ifigure);

%  N-list  V0  P0  B  BP  B2P  error (direct)
%    1     2   3   4   5   6   7


compare_res=[fitted_res(1,:); pvfit_res(1,:); fitted_res(2,:); pvfit_res(2,:); fitted_res(3,:); pvfit_res(3,:); ...
           fitted_res(4,:); pvfit_res(4,:)] 
final_res=[fitted_res; pvfit_res]
end
if ipress < 0 ;  final_res=fitted_res, end

ff=['out_eosres'];   eval(['save ' ff ' final_res -ascii']); 

mm=load('magtot.0'); 

if length(mm) > 1; mm=mm(datarange); figure, plot(volume, mm, '-o'); title ('magnetic moment');
saveas(gcf, 'fig2_V_mm.png') 
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%% END of MAIN %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function res=build_eos(prop, L)

%  V0  E0  P0  B  BP  B2P  av_diff   max_diff                                                            
%  1   2   3   4  5   6     7        8                                                               

V    = prop(1);          %  V0(A^3/atom) 
E0   = prop(2);          %  B0(GPa)   
B    = prop(4);          %  BP  
bp   = prop(5);          %  E0  (eV/atom) 
b2p  = prop(6);          %  b2p  (1/GPa)

changeunit = 1.0/160.21892;         %  1 GPa = 1/160.22 eV/A^3 
B          = B*changeunit;	     %  (nx1) B0 (eV/A^3) 
b2p        = b2p/changeunit; % A^3/eV
%
if L==1 | L==2

a=(8*E0 + 3*B*(122 + 9*B*b2p - 57*bp + 9*bp^2)*V)/8;
e=(3*B*(74 + 9*B*b2p - 45*bp + 9*bp^2)*V^(7/3))/8;
d=(-3*B*(83 + 9*B*b2p - 48*bp + 9*bp^2)*V^2)/2;
c=(9*B*(94 + 9*B*b2p - 51*bp + 9*bp^2)*V^(5/3))/4;
b=(-3*B*(107 + 9*B*b2p - 54*bp + 9*bp^2)*V^(4/3))/2;

if abs(e) < 1e-8, e=0; end    
res=[a, b, c, d, e];      

end

if L==3 | L==4
a=(128*E0 + 3*B*(287 + 9*B*b2p - 87*bp + 9*bp^2)*V)/128;
e=(3*B*(143 + 9*B*b2p - 63*bp + 9*bp^2)*V^(11/3))/128;
d=(-3*B*(167 + 9*B*b2p - 69*bp + 9*bp^2)*V^3)/32;
c=(9*B*(199 + 9*B*b2p - 75*bp + 9*bp^2)*V^(7/3))/64;
b=(-3*B*(239 + 9*B*b2p - 81*bp + 9*bp^2)*V^(5/3))/32;

if abs(e) < 1e-8, e=0; end    
res=[a, b, c, d, e];      

end

if L==5 | L==6
a=(24*E0 + 12*B*V*log(V)^2 + 4*B*(-2 + bp)*V*log(V)^3 + ...
      B*(3 + B*b2p - 3*bp + bp^2)*V*log(V)^4)/24;
b=-(B*V*log(V)*(6 + 3*(-2 + bp)*log(V) + (3 + B*b2p - 3*bp + bp^2)*log(V)^2))/6;
c=(B*V*(2 + 2*(-2 + bp)*log(V) + (3 + B*b2p - 3*bp + bp^2)*log(V)^2))/4;
d=-(B*V*(-2 + bp + (3 + B*b2p - 3*bp + bp^2)*log(V)))/6;
e=(B*(3 + B*b2p - 3*bp + bp^2)*V)/24;

if abs(e) < 1e-8, e=0; end    
res=[a, b, c, d, e];      

end

end

%%%%%%%%%%%%%############
%%%%  prop = eosparameter45(xx, vzero, icase); % [V0, E0, P0, B, BP, B2P];
function res=eosparameter45(xx, vzero, icase) 

 chunit=160.2189;
 a=xx(1);  b=xx(2);  c=xx(3);  d=xx(4);  e=xx(5);
%%%% to get the structure properties%%%%%%%%%%%%
if icase == 1
   V = 4*c^3 - 9*b*c*d + sqrt((c^2 - 3*b*d)*(4*c^2 -3*b*d)^2); V =-V/b^3;
end

if icase == 2
  fun=@(x)((4*e)/(3*x^(7/3)) + d/x^2 + (2*c)/(3*x^(5/3)) + b/(3*x^(4/3)))*chunit;
  V=fzero(fun,vzero);
end

if icase ==1 | icase == 2

P=(4*e)/(3*V^(7/3)) + d/V^2 + (2*c)/(3*V^(5/3)) + b/(3*V^(4/3)); P=P*chunit; 

B = ((28*e)/(9*V^(10/3)) + (2*d)/V^3 + (10*c)/(9*V^(8/3)) + (4*b)/(9*V^(7/3)))*V;  
B = B*chunit;

BP=(98*e + 54*d*V^(1/3) + 25*c*V^(2/3) + 8*b*V)/(42*e + ... 
    27*d*V^(1/3) + 15*c*V^(2/3) + 6*b*V); 

B2P =(V^(8/3)*(9*d*(14*e + 5*c*V^(2/3) + 8*b*V) + ... 
     2*V^(1/3)*(126*b*e*V^(1/3) + 5*c*(28*e + b*V))))/(2*(14*e + ... 
     9*d*V^(1/3) + 5*c*V^(2/3) + 2*b*V)^3);
B2P = B2P/chunit; 

E0=a + b*V^(-1/3) + c*V^(-2/3)+ d*V^(-1) + e*V^(-4/3); 

res=[V, E0, P, B, BP, B2P];

end
%%%%%%%%%%%%%%%%%%%%

if icase == 3
   V =sqrt(-((4*c^3 - 9*b*c*d + sqrt((c^2 - 3*b*d)*(4*c^2 - 3*b*d)^2))/b^3));
end

if icase == 4
  fun=@(x)(((8*e)/(3*x^(11/3)) + (2*d)/x^3 + (4*c)/(3*x^(7/3)) + ...
             (2*b)/(3*x^(5/3)))*chunit);
  V=fzero(fun,vzero);
end

if icase ==3 | icase == 4

P=(8*e)/(3*V^(11/3)) + (2*d)/V^3 + (4*c)/(3*V^(7/3)) + (2*b)/(3*V^(5/3)); P=P*chunit; 

B =(2*(44*e + 27*d*V^(2/3) + 14*c*V^(4/3) + 5*b*V^2))/(9*V^(11/3));
B = B*chunit;

BP=(484*e + 243*d*V^(2/3) + 98*c*V^(4/3) + 25*b*V^2)/(132*e + ... 
     81*d*V^(2/3) + 42*c*V^(4/3) + 15*b*V^2);

B2P =(4*V^(13/3)*(27*d*(22*e + 7*c*V^(4/3) + 10*b*V^2) + V^(2/3)*(990*b*e*V^(2/3) + ... 
  7*c*(176*e + 5*b*V^2))))/(44*e + 27*d*V^(2/3) + 14*c*V^(4/3) + 5*b*V^2)^3;

B2P = B2P/chunit; 

E0= a + e/V^(8/3) + d/V^2 + c/V^(4/3) + b/V^(2/3);

res=[V, E0, P, B, BP, B2P];

end
%%%%%%%%%%%%%%%%%%%%

if icase ==5 | icase == 6
    
 fun=@(x)((-((b + 2*c*log(x) + 3*d*log(x)^2 + 4*e*log(x)^3)/x))*chunit);
 V=fzero(fun,vzero);
 
P= -((b + 2*c*log(V) + 3*d*log(V)^2 + 4*e*log(V)^3)/V);  P=P*chunit; 

B =-((b - 2*c + 2*(c - 3*d)*log(V) + ...
 3*(d - 4*e)*log(V)^2 + 4*e*log(V)^3)/V); 
B = B*chunit;

BP=(b - 4*c + 6*d + 2*(c - 6*d + 12*e)*log(V) + ... 
 3*(d - 8*e)*log(V)^2 + 4*e*log(V)^3)/(b - 2*c + ... 
 2*(c - 3*d)*log(V) + 3*(d - 4*e)*log(V)^2 + 4*e*log(V)^3);

B2P =(2*V*(2*c^2 - 3*b*d + 18*d^2 + 12*b*e - 6*c*(d + 4*e) + ...
 6*(c*d - 3*d^2 - 2*b*e + 12*d*e)*log(V) + 9*(d - 4*e)^2*log(V)^2 + ...
 24*(d - 4*e)*e*log(V)^3 + 24*e^2*log(V)^4))/(b - 2*c + 2*(c - 3*d)*log(V) + ...
 3*(d - 4*e)*log(V)^2 + 4*e*log(V)^3)^3;

B2P = B2P/chunit; 

E0= a + b*log(V) + c*log(V)^2 + d*log(V)^3 + e*log(V)^4;

res=[V, E0, P, B, BP, B2P];

end
%%%%%%%%%%%%%
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
 function res=morseeos(xini,Data)
%  
%  Equation of state: Morse 
%  % V-1      E0-2       B-3        bp-4
%
%
 V  = xini(1);
 E0 = xini(2);
 B  = xini(3);
 bp = xini(4);
%
   a= E0 + (9*B*V)/(2*(-1 + bp)^2);
   b= (-9*B*exp(-1 + bp)*V)/(-1 + bp)^2;
   c= (9*B*exp(-2 + 2*bp)*V)/(2*(-1 + bp)^2);
   d= (1 - bp)/V^(1/3);

x=Data(:,1);  % volume
y=Data(:,2);  % energy

eng = a + b.*exp(d.*x.^(1/3)) + c.*exp(2*d.*x.^(1/3)); 
res = eng - y;

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
 function res=murnaghan(xini,Data)
%  
%  Equation of state: Murnaghan  
%  % V-1      E0-2       B-3        bp-4
%
%
 V  = xini(1);
 E0 = xini(2);
 B  = xini(3);
 bp = xini(4);
%
 x=Data(:,1);
 y=Data(:,2);

 eng = E0 - (B*V)./(-1 + bp)+(B.*(1+(V./x).^bp./(-1+bp)).*x)./bp; 
 res=eng-y;

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function res=pveosfit(volume, fzero, pressure, vv, isave, ifigure)

 numbereos=4;        
 if numbereos==4;  ieos=[1 2 3 4];  end
 
 chunit=160.2189;
 vzero=mean(volume); 
 res=[];  resdiff=[]; resxx=[]; respp=[];
%

for L = ieos 
%%%%%% 
if L==1   %%%   mBM4 
   
   qq=volume;
   bb=pressure;
   AA=[qq.^(-4/3)*(1/3), qq.^(-5/3)*(2/3), qq.^(-2)];  
   xx=pinv(AA)*bb;   %(4x1)=(4xn)*(nx1), solve it by pseudo-inversion: Ax=b
   b=xx(1);  c=xx(2);  d=xx(3); e=0.0;  xx=[b c d e];

 pp= (4*e)./(3*vv.^(7/3)) + d./vv.^2 + (2*c)./(3*vv.^(5/3)) + b./(3*vv.^(4/3));  
 pp0= (4*e)./(3*volume.^(7/3)) + d./volume.^2 + (2*c)./(3*volume.^(5/3)) + b./(3*volume.^(4/3)); 

 if ifigure >1; figure, plot(vv,pp, volume, pressure, 'o'), title('P-V FITTED curve, mBM4, No. 1, GPa'), end
 diffp=pp0-pressure;
 
 prop = pv2prop(xx, vzero, L); % [V0, P, B, BP, B2P];
 vzero=prop(1);
 
 newxx   =pvbuildeos(prop, L);
 resxx   =[resxx; L, xx; L, newxx];
 
 nnn=length(bb);
 qwe=(diffp).^2; 
 asd=sqrt(sum(qwe/nnn)); 
 res     =[res; L, prop, asd];
 respp=[respp; pp];
 resdiff =[resdiff, diffp]; 
 
end % end of L==1 
 
%%%%%% 
if L == 2  %% mBM5 
      
 qq=volume;
 bb=pressure;
 AA=[qq.^(-4/3)*(1/3), qq.^(-5/3)*(2/3), qq.^(-2), qq.^(-7/3)*(4/3)];  
 xx=pinv(AA)*bb;   %(4x1)=(4xn)*(nx1), solve it by pseudo-inversion: Ax=b
 b=xx(1);  c=xx(2);  d=xx(3); e=xx(4);  xx=[b c d e];
 
 pp=(4*e)./(3*vv.^(7/3)) + d./vv.^2 + (2*c)./(3*vv.^(5/3)) + b./(3*vv.^(4/3));  
 pp0= (4*e)./(3*volume.^(7/3)) + d./volume.^2 + (2*c)./(3*volume.^(5/3)) + b./(3*volume.^(4/3)); 
 if ifigure >1; figure, plot(vv,pp, volume, pressure, 'o'), title('P-V FITTED curve, mBM5, No. 2, GPa'), end
 diffp=pp0-pressure;
 
 prop =pv2prop(xx, vzero, L); % [V0, P, B, BP, B2P];
 
 newxx   =pvbuildeos(prop, L);
 resxx   =[resxx; L, xx; L, newxx];
  
 nnn=length(bb);
 qwe=(diffp).^2; 
 asd=sqrt(sum(qwe/nnn)); 
 res     =[res; L, prop, asd];
 respp=[respp; pp];
 resdiff =[resdiff, diffp]; 
end % end of L==2 

%%%%%% 
if L == 3  %% BM4 
    
 qq=volume;
 bb=pressure;
 AA=[qq.^(-5/3)*(2/3), qq.^(-7/3)*(4/3), qq.^(-3)*2]; 
 xx=pinv(AA)*bb;   %(4x1)=(4xn)*(nx1), solve it by pseudo-inversion: Ax=b
 b=xx(1);  c=xx(2);  d=xx(3); e=0.0;  xx=[b c d e];
 
 x=vv;
 pp=(8*e)./(3*x.^(11/3))+(2*d)./x.^3+(4*c)./(3*x.^(7/3))+(2*b)./(3*x.^(5/3));
 x=volume;
 pp0=(8*e)./(3*x.^(11/3))+(2*d)./x.^3+(4*c)./(3*x.^(7/3))+(2*b)./(3*x.^(5/3));
 if ifigure >1; figure, plot(vv,pp, volume, pressure, 'o'), title('P-V FITTED curve, BM4, No. 3, GPa'), end
 diffp=pp0-pressure;
 
 prop= pv2prop(xx, vzero, L); % [V0, P, B, BP, B2P]; 
 bm4_v0=prop(1); bm4_B0=prop(3); bm4_Bp=prop(4); 

 newxx   =pvbuildeos(prop, L);
 resxx   =[resxx; L, xx; L, newxx];
 
 nnn=length(bb);
 qwe=(diffp).^2; 
 asd=sqrt(sum(qwe/nnn)); 
 res     =[res; L, prop, asd];
 respp=[respp; pp];
 resdiff =[resdiff, diffp]; 
end % end of L==3 

%%%%%% 
if L == 4  %% BM5
     
  qq=volume;
  bb=pressure;
  AA=[qq.^(-5/3)*(2/3), qq.^(-7/3)*(4/3), qq.^(-3)*2, qq.^(-11/3)*(8/3)]; 
  xx=pinv(AA)*bb;   %(4x1)=(4xn)*(nx1), solve it by pseudo-inversion: Ax=b
  b=xx(1);  c=xx(2);  d=xx(3); e=xx(4);  xx=[b c d e];
 
 x=vv;
 pp=(8*e)./(3*x.^(11/3))+(2*d)./x.^3+(4*c)./(3*x.^(7/3))+(2*b)./(3*x.^(5/3));
 x=volume;
 pp0=(8*e)./(3*x.^(11/3))+(2*d)./x.^3+(4*c)./(3*x.^(7/3))+(2*b)./(3*x.^(5/3));
 if ifigure >1; figure, plot(vv,pp, volume, pressure, 'o'), title('P-V curve, BM5, No. 4, GPa'), end
 diffp=pp0-pressure;

 prop= pv2prop(xx, vzero, L); % [V0, P, B, BP, B2P];
  
 newxx   =pvbuildeos(prop, L);
 resxx   =[resxx; L, xx; L, newxx];
 
 nnn=length(bb);
 qwe=(diffp).^2; 
 asd=sqrt(sum(qwe/nnn)); 
 res     =[res; L, prop, asd];
 respp=[respp; pp];
 resdiff =[resdiff, diffp]; 
end % end of L==4 


%%%%%% 
end % end of for L=ieos

if numbereos==4;
pp4=[respp(1,:); respp(3,:)]; 
pp5=[respp(2,:); respp(4,:)]; 
figure, plot(vv, pp4, vv, pp5, '--', volume, pressure,'o'), 
textPV = ['P-V fitted: BM4 V-B-Bp = ', string(bm4_v0), ' ', string(bm4_B0), ' ', string(bm4_Bp)];
title(join(textPV)), 
legend('1-mBM4', '3-BM4', '2-mBM5','4-BM5');
saveas(gcf, 'fig40_PV_fitted.png')
end

if numbereos==2;
pp4=respp(1,:); 
pp5=respp(2,:); 
figure, plot(vv, pp4, vv, pp5, '--', volume, pressure,'o'), 
textPV = ['P-V fitted: BM4 V-B-Bp = ', string(bm4_v0), ' ', string(bm4_B0), ' ', string(bm4_Bp)];
title(join(textPV))
legend('1-mBM4', '2-mBM5');
saveas(gcf, 'fig20_PV_fitted.png') 
end


resxx;

max_diff= max(abs(resdiff));
av_diff =mean(abs(resdiff));
dpp_av_max=[av_diff; max_diff];

pvfit_res=res; %[res(:,1:2), res(:,4:end)]

%ff=['outpv_eosres'];   eval(['save ' ff ' pvfit_res -ascii']); 


if isave > 0;  
 resvp =[vv', respp']; 
 ff=['pvfit_VP'];  eval(['save ' ff ' resvp -ascii']); 
end

end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
 function res=vineteos(xini,Data)
%  
%  Equation of state: Murnaghan  
%  % V-1      E0-2       B-3        bp-4
%
%
 V  = xini(1);
 E0 = xini(2);
 B  = xini(3);
 bp = xini(4);
%
x=Data(:,1);
y=Data(:,2);

eng =E0 + (4*B*V)./(-1 + bp).^2 - (4*B*V*(1 + (3*(-1 + bp)*(-1 + ...
    (x/V).^(1/3)))/2))./((-1 + bp).^2.*exp((3*(-1 + bp).*(-1 + (x./V).^(1/3)))/2));
res = eng - y;
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function res=pv2prop(xx, vzero, icase) 

 chunit=160.2189;
 b=xx(1);  c=xx(2);  d=xx(3);  e=xx(4);
%%%% to get the structure properties%%%%%%%%%%%%
if icase == 1
   V = 4*c^3 - 9*b*c*d + sqrt((c^2 - 3*b*d)*(4*c^2 -3*b*d)^2); V =-V/b^3;
end

if icase == 2
  fun=@(x)((4*e)/(3*x^(7/3)) + d/x^2 + (2*c)/(3*x^(5/3)) + b/(3*x^(4/3)));
  V=fzero(fun,vzero);
end

if icase ==1 | icase == 2

P=(4*e)/(3*V^(7/3)) + d/V^2 + (2*c)/(3*V^(5/3)) + b/(3*V^(4/3));  

B = ((28*e)/(9*V^(10/3)) + (2*d)/V^3 + (10*c)/(9*V^(8/3)) + (4*b)/(9*V^(7/3)))*V;  


BP=(98*e + 54*d*V^(1/3) + 25*c*V^(2/3) + 8*b*V)/(42*e + ... 
    27*d*V^(1/3) + 15*c*V^(2/3) + 6*b*V); 

B2P =(V^(8/3)*(9*d*(14*e + 5*c*V^(2/3) + 8*b*V) + ... 
     2*V^(1/3)*(126*b*e*V^(1/3) + 5*c*(28*e + b*V))))/(2*(14*e + ... 
     9*d*V^(1/3) + 5*c*V^(2/3) + 2*b*V)^3);


res=[V, P, B, BP, B2P];

end
%%%%%%%%%%%%%%%%%%%%

if icase == 3
   V =sqrt(-((4*c^3 - 9*b*c*d + sqrt((c^2 - 3*b*d)*(4*c^2 - 3*b*d)^2))/b^3));
end

if icase == 4
  fun=@(x)(((8*e)/(3*x^(11/3)) + (2*d)/x^3 + (4*c)/(3*x^(7/3)) + ...
             (2*b)/(3*x^(5/3)))*chunit);
  V=fzero(fun,vzero);
end

if icase ==3 | icase == 4

P=(8*e)/(3*V^(11/3)) + (2*d)/V^3 + (4*c)/(3*V^(7/3)) + (2*b)/(3*V^(5/3)); 

B =(2*(44*e + 27*d*V^(2/3) + 14*c*V^(4/3) + 5*b*V^2))/(9*V^(11/3));


BP=(484*e + 243*d*V^(2/3) + 98*c*V^(4/3) + 25*b*V^2)/(132*e + ... 
     81*d*V^(2/3) + 42*c*V^(4/3) + 15*b*V^2);

B2P =(4*V^(13/3)*(27*d*(22*e + 7*c*V^(4/3) + 10*b*V^2) + V^(2/3)*(990*b*e*V^(2/3) + ... 
  7*c*(176*e + 5*b*V^2))))/(44*e + 27*d*V^(2/3) + 14*c*V^(4/3) + 5*b*V^2)^3;


res=[V, P, B, BP, B2P];

end
%%%%%%%%%%%%%%%%%%%%
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function res=pvbuildeos(prop, L)

%  V0  P0  B  BP  B2P  av_diff   max_diff                                                            
%  1   2   3   4  5     6           7
V    = prop(1);          %  V0(A^3/atom) 
P0   = prop(2);          %  P0(GPa)   
B    = prop(3);          %  B  GPa
bp   = prop(4);          %  BP  
b2p  = prop(5);          %  b2p  (1/GPa)

%changeunit = 1.0/160.21892;         %  1 GPa = 1/160.22 eV/A^3 
%B          = B*changeunit;	     %  (nx1) B0 (eV/A^3) 
%b2p        = b2p/changeunit; % A^3/eV

if L==1 | L==2
e=(3*B*(74 + 9*B*b2p - 45*bp + 9*bp^2)*V^(7/3))/8;
d=(-3*B*(83 + 9*B*b2p - 48*bp + 9*bp^2)*V^2)/2;
c=(9*B*(94 + 9*B*b2p - 51*bp + 9*bp^2)*V^(5/3))/4;
b=(-3*B*(107 + 9*B*b2p - 54*bp + 9*bp^2)*V^(4/3))/2;

if abs(e) < 1e-8, e=0; end    
res=[b, c, d, e];      

end


if L==3 | L==4

e=(3*B*(143 + 9*B*b2p - 63*bp + 9*bp^2)*V^(11/3))/128;
d=(-3*B*(167 + 9*B*b2p - 69*bp + 9*bp^2)*V^3)/32;
c=(9*B*(199 + 9*B*b2p - 75*bp + 9*bp^2)*V^(7/3))/64;
b=(-3*B*(239 + 9*B*b2p - 81*bp + 9*bp^2)*V^(5/3))/32;

if abs(e) < 1e-8, e=0; end    
res=[b, c, d, e];      
end

end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

