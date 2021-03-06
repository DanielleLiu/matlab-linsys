addpath(genpath('../../'))
%% Create model:
D1=5;
D2=200;%100;
%CS 2006 gets progressively slower for larger D2 (linear execution time with D2 for large D2).
%This implementation grows linearly too but with the SMALLEST of D1,D2. For
%small D2, CS2006 is slightly faster, as it does not enforce covariance
%matrices to be PSD. This sometimes results in ugly filtering (especially
%with large covariance matrices, the smoothing does not work well, even
%being less accurate than this implementation's filtering).
A=diag(rand(D1,1));
A=.9999*A; %Setting the max eigenvalue to .9999
%A(1,1)=.9999; %Very slow pole. This renders fast mode basically useless, and thus makes (reduced) C code much faster than any matlab code (steady state filtering can still be forced, but there will be a trade-off with speed, especially in the determination of the very slow state)
%B=3*randn(D1,1);
%B=B./sign(B); %Forcing all elements of B to be >0, WLOG
B=(eye(size(A))-A)*ones(size(A,1),1); %WLOG, arbitrary scaling so all states asymptote at 1
N=2000;
U=[zeros(300,1);ones(N,1);zeros(N/2,1)]'; %Step input and then removed
C=randn(D2,D1);
D=randn(D2,1);
Q=eye(D1)*1e-1;
%Q=zeros(D1); %Uncomment to check performance with Q=0
R=1*eye(D2); %CS2006 performance degrades (larger state estimation errors) for very small R

%% Simulate
NN=size(U,2);
x0=zeros(D1,1);
[Y,X]=fwdSim(U,A,B,C,D,x0,Q,R);
[Y1,X1]=fwdSim(U,A,B,C,D,x0,[],[]); %Noiseless simulation, for comparison
%Y(:,400)=NaN;
X=X(:,1:end-1);

%% Define initial uncertainty
P0=zeros(D1);
P0=diag(Inf*ones(D1,1));
P0=1e1*eye(D1);

%% Do kalman smoothing with true params, no fast mode, no reduction
tic
mdl=1;
opts.noReduceFlag=true;
opts.fastFlag=0;
[Xs,Ps,Pt,Xf,Pf,Xp,Pp,~,logL(mdl)]=statKalmanSmoother(Y,A,C,Q,R,x0,P0,B,D,U,opts); %Kalman smoother estimation of states, given the true parameters (this is the best possible estimation of states)
Xcs=Xs;
Pcs=Ps;
res(mdl)=norm(Xs-X,'fro')^2;
maxRes(mdl)=max(sum((Xs-X).^2));
name{mdl}='KS';
res2KS(mdl)=norm(Xs-Xcs,'fro');
tc(mdl)=toc;

%% Do kalman smoothing with true params, no fast mode, reduction
tic
mdl=2;
opts.noReduceFlag=false;
opts.fastFlag=0;
[Xs,Ps,Pt,Xf,Pf,Xp,Pp,~,logL(mdl)]=statKalmanSmoother(Y,A,C,Q,R,x0,P0,B,D,U,opts); %Kalman smoother estimation of states, given the true parameters (this is the best possible estimation of states)
res(mdl)=norm(Xs-X,'fro')^2;
maxRes(mdl)=max(sum((Xs-X).^2));
name{mdl}='KS red';
res2KS(mdl)=norm(Xs-Xcs,'fro');
tc(mdl)=toc;

%% Do kalman smoothing with true params, fast mode, reduction
tic
mdl=3;
opts.noReduceFlag=false;
opts.fastFlag=1;
[Xs,Ps,Pt,Xf,Pf,Xp,Pp,~,logL(mdl)]=statKalmanSmoother(Y,A,C,Q,R,x0,P0,B,D,U,opts); %Kalman smoother estimation of states, given the true parameters (this is the best possible estimation of states)
res(mdl)=norm(Xs-X,'fro')^2;
maxRes(mdl)=max(sum((Xs-X).^2));
name{mdl}='KSred fast';
res2KS(mdl)=norm(Xs-Xcs,'fro');
tc(mdl)=toc;

%% Use Info smoother:
mdl=4;
tic;
[is,Is,iif,If,ip,Ip,Xs,Ps,~]=statInfoSmoother2(Y,A,C,Q,R,x0,P0,B,D,U,opts);
res(mdl)=norm(Xs-X,'fro')^2;
maxRes(mdl)=max(sum((Xs-X).^2));
logL(mdl)=nan;
name{mdl}='IS';
res2KS(mdl)=norm(Xs-Xcs,'fro');
tc(mdl)=toc;

%% Use Cheng & Sabes code
[folder]=fileparts(mfilename('fullpath'));
addpath(genpath([folder '/../../ext/lds-1.1/']))

%%(matlab version), reduced:
mdl=5;
opts.noReduceFlag=false;
tic
[Xs,Ps,Pt,logL(mdl)]=statKalmanSmootherCS2006Matlab(Y,A,C,Q,R,x0,P0,B,D,U,opts);
res(mdl)=norm(Xs-X,'fro')^2;
maxRes(mdl)=max(sum((Xs-X).^2));
name{mdl}='CSred';
res2KS(mdl)=norm(Xs-Xcs,'fro');
tc(mdl)=toc;

%% C version, reduced:
if isempty(which('SmoothLDS')) %Function not found
    cd ../../ext/lds-1.1/
    mex utils.c gsl_utils.c SmoothLDS.c -silent -output SmoothLDS -largeArrayDims  %Compiling from source
    cd ../../kalman/test/ %This directory
end
mdl=6;
opts.noReduceFlag=false; %C version is very slow for systems with large number of inputs, but very fast for systems with a low number
tic
[Xs,Ps,Pt,logL(mdl)]=statKalmanSmootherCS2006(Y,A,C,Q,R,x0,P0,B,D,U,opts);
res(mdl)=norm(Xs-X,'fro')^2;
maxRes(mdl)=max(sum((Xs-X).^2));
name{mdl}='CSred C';
res2KS(mdl)=norm(Xs-Xcs,'fro');
tc(mdl)=toc;

%% CS, Matlab version, no reduce (C version is terribly slow for large problems)
mdl=7;
opts.noReduceFlag=true;
tic
[Xs,Ps,Pt,logL(mdl)]=statKalmanSmootherCS2006Matlab(Y,A,C,Q,R,x0,P0,B,D,U,opts);
res(mdl)=norm(Xs-X,'fro')^2;
maxRes(mdl)=max(sum((Xs-X).^2));
name{mdl}='CS';
res2KS(mdl)=norm(Xs-Xcs,'fro');
tc(mdl)=toc;
%%
%save results.mat res tc logL maxRes res2KS name A
%% Visualize results
%load results.mat
%name={'KS','KSred','KSredF','IS','CSred','CSred C','CS'};
idx=1:7;
figure('Units','Pixels','InnerPosition',[100 100 300*4 300])
subplot(1,4,1) %Bar of residuals
bar(sqrt(res(idx)),'EdgeColor','none')
set(gca,'XTickLabel',name(idx),'YScale','log')
title('RMSE residuals')
axis tight
aa=axis;
axis([aa(1:2) min(sqrt(res(idx)))*.9999 sqrt(max(res(1:end-1)))*1.0001])
grid on

subplot(1,4,2) %Bar of residuals to KS
bar(res2KS(idx),'EdgeColor','none')
set(gca,'XTickLabel',name(idx))
title('RMSE residuals to KS')
axis tight
aa=axis;
axis([aa(1:2) 0 1e-12])
grid on

subplot(1,4,3) %Bar of max res
bar(sqrt(maxRes(idx)),'EdgeColor','none')
set(gca,'XTickLabel',name(idx))
title('Max sample rMSE')
axis tight
aa=axis;
axis([aa(1:2) 0 sqrt(max(maxRes(idx)))*1.1])
grid on

%subplot(1,5,4) %Bar of logL
%bar(logL(idx),'EdgeColor','none')
%set(gca,'XTickLabel',name(idx))
%title('LogL')
%axis tight
%aa=axis;
%axis([aa(1:2) logL(1)+[-1 1]*1e-5])
%grid on

subplot(1,4,4) %Bar of running times
bar(tc(idx)/tc(3),'EdgeColor','none')
set(gca,'XTickLabel',name(idx),'YScale','log','YTick',[.5 1 1e1 1e2 1e3])
axis tight
aa=axis;
axis([aa(1:2) .5 2e2])
title(['Relative running time(s), KSred fast=' num2str(tc(3),3)])
grid on

logL
%Save fig
%print(fh,['results.eps'],'-depsc','-tiff','-r2400')

%% Just time
load results.mat
name={'KS','KSred','KSred_f','IS','CSred','CSred C','CS'};
idx=[1,2,3,7];
figure('Units','Pixels','InnerPosition',[100 100 300*2 300])
bar(tc(idx)/tc(3),'EdgeColor','none','FaceAlpha',.7)
set(gca,'XTickLabel',name(idx),'YScale','log','YTick',[.5 1 1e1 1e2 1e3])
axis tight
aa=axis;
axis([0 5 .5 2e2])
title(['Smoothing time comparison'])
text(2.67,1.3,[ num2str(tc(3),3) ' s'])
ylabel('Relative running time')

grid on
