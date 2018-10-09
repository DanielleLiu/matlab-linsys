%%
A=[.999,0; 0 , .95];
B=[.001; .05]; %Both states asymptote at 1
Ny=3;
C=randn(Ny,2);
D=randn(Ny,1);
Q=zeros(2);
R=.01*eye(Ny);
x0=[0;0];
U=[zeros(1,500), ones(1,1000)];
d=2;
%%
Niter=2e1;
err1=nan(size(A,1),Niter);
err_=nan(size(A,1),Niter);
err__=nan(size(A,1),Niter);
for k=1:Niter
%% Sim:
[Y,X]=fwdSim(U,A,B,C,D,x0,Q,R);

%% Subspace ID: (traditional)
[A1,B1,C1,D1,X1,Q1,R1]=subspaceID(Y,U,d);
err1(:,k)=sort(eig(A1))-sort(eig(A));
logLperSamplePerDim=dataLogLikelihood(Y,U,A1,B1,C1,D1,Q1,R1)
%%
[A_,B_,C_,D_,X_,Q_,R_]=subspaceIDv2(Y,U,d);
err_(:,k)= sort(eig(A_))-sort(eig(A));
logLperSamplePerDim=dataLogLikelihood(Y,U,A_,B_,C_,D_,Q_,R_)
opts.Niter=100;
[Xs,Ps,Pt,Xf,Pf,Xp,Pp,rejSamples]=statKalmanSmoother(Y,A_,C_,Q_,R_,[],[],B_,D_,U);
[~,~,~,~,~,~,~,~,bestLogL]=EM(Y,U,Xs,opts,Ps);

%%
%[A__,B__,C__,D__,X__,Q__,R__]=subspaceIDv3(Y,U,d);
%err__(:,k)= sort(eig(A__))-sort(eig(A));
end
%%
mean(err1,2)
std(err1,[],2)
mean(err_,2)
std(err_,[],2)
mean(err__,2)
std(err__,[],2)
%eig(A)
%eig(A1)
%trace(R1)
%eig(A_)
%trace(R_)