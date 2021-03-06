%% Create model:
D1=2;
D2=1;
N=1000;
A=diag(rand(D1,1));
A=.9999*A; %Setting the max eigenvalue to .9999
%B=3*randn(D1,1);
%B=B./sign(B); %Forcing all elements of B to be >0, WLOG
B=(eye(size(A))-A)*ones(size(A,1),1); %WLOG, arbitrary scaling so all states asymptote at 1
U=[zeros(300,1);ones(N,1);zeros(N/2,1)]'; %Step input and then removed
C=randn(D2,D1);
D=randn(D2,1);
Q=eye(D1)*1e-3;
R=1*eye(D2); %CS2006 performance degrades (larger state estimation errors) for very small R

%% Simulate
NN=size(U,2);
x0=zeros(D1,1);
[Y,X]=fwdSim(U,A,B,C,D,x0,Q,R);
[Y1,X1]=fwdSim(U,A,B,C,D,x0,[],[]); %Noiseless simulation, for comparison

%% Do kalman smoothing with true params
fastFlag=[];
opts.fastFlag=0;
opts.noReduceFlag=false;
[Xf,Pf,Xp,Pp,rejSamples,logL]=statKalmanFilter(Y,A,C,Q,R,[],[],B,D,U,opts); %Kalman smoother estimation of states, given the true parameters (this is the best possible estimation of states)
tf=timeit(@() statKalmanFilter(Y,A,C,Q,R,[],[],B,D,U,opts));
%% Kalman classic with no reduce
fastFlag=[];
opts.fastFlag=0;
opts.noReduceFlag=true;
[XfNR,~,~,~,~,logLNR]=statKalmanFilter(Y,A,C,Q,R,[],[],B,D,U,opts); %Kalman smoother estimation of states, given the true parameters (this is the best possible estimation of states)
tfNR=timeit(@() statKalmanFilter(Y,A,C,Q,R,[],[],B,D,U,opts));
%% Use Info smoother: (there is no reduction here ever)
%[Xcs,Pcs,Xp,Pp,rejSamples,logLInfo]=statInfoFilter(Y,A,C,Q,R,[],[],B,D,U,opts); 
[ii,I,ip,Ip,Xcs,Pcs,logLInfo]=statInfoFilter2(Y,A,C,Q,R,[],[],B,D,U,opts); 
tcs=timeit(@() statInfoFilter2(Y,A,C,Q,R,[],[],B,D,U,opts));
%tcs2=timeit(@() statInfoFilter2(Y,A,C,Q,R,[],[],B,D,U,opts)); %This filter does not compute x,P by default, just i and I
%% Use Info smoother: (there is no reduction here ever)
opts.fastFlag=false;
opts.noReduceFlag=false;
[Xsqrt,Pcs,rejSamples,logLsqrt]=statSqrtFilter(Y,A,C,Q,R,[],[],B,D,U,opts); 
tsqrt=timeit(@() statSqrtFilter(Y,A,C,Q,R,[],[],B,D,U,opts));
%% Visualize results
figure
for i=1:2
    subplot(3,1,i)
    hold on
    plot(Xf(i,:),'DisplayName','Filtered','LineWidth',2)
    plot(XfNR(i,:),'DisplayName','Filtered NR','LineWidth',2)
    plot(Xcs(i,:),'DisplayName','InfoFiltered','LineWidth',2)
    plot(Xsqrt(i,:),'DisplayName','SQRTFiltered','LineWidth',2)
    plot(X(i,:),'DisplayName','Actual','LineWidth',2)

    legend
    if i==1
        title(['This runtime= ' num2str(tf) ', NoReduce runtime= ' num2str(tfNR) ', Info runtime= ' num2str(tcs) ', Sqrt runtime= ' num2str(tsqrt)]);
    elseif i==2
        title(['This log-L= ' num2str(logL) ', NoReduce log-L= ' num2str(logLNR)  ', Info log-L= ' num2str(logLInfo) ', Sqrt log-L= ' num2str(logLsqrt)]);
    end
end
subplot(3,1,3)
for i=1:2
    hold on
    set(gca,'ColorOrderIndex',1)
    p(1)=plot(Xf(i,:)-X(i,1:end-1));
     p(2)=plot(XfNR(i,:)-X(i,1:end-1));
    p(3)=plot(Xcs(i,:)-X(i,1:end-1));
        p(4)=plot(Xsqrt(i,:)-X(i,1:end-1));

    set(gca,'ColorOrderIndex',1)
    aux=sqrt(mean((X(i,1:end-1)-Xf(i,:)).^2));
    b1=bar(1900+400*i,aux,'BarWidth',100,'EdgeColor','None');
    text(1850+400*i,1.2*aux,num2str(aux,3),'Color',b1.FaceColor)
    
        aux=sqrt(mean((X(i,1:end-1)-XfNR(i,:)).^2));
    b1=bar(2000+400*i,aux,'BarWidth',100,'EdgeColor','None');
    text(1950+400*i,1.2*aux,num2str(aux,3),'Color',b1.FaceColor)
    
    aux=sqrt(mean((X(i,1:end-1)-Xcs(i,:)).^2));
    b1=bar(2100+400*i,aux,'BarWidth',100,'EdgeColor','None');
    text(2050+400*i,1.4*aux,num2str(aux,3),'Color',b1.FaceColor)
    grid on
    
      aux=sqrt(mean((X(i,1:end-1)-Xsqrt(i,:)).^2));
    b1=bar(2200+400*i,aux,'BarWidth',100,'EdgeColor','None');
    text(2150+400*i,1.4*aux,num2str(aux,3),'Color',b1.FaceColor)
    grid on
end
title('Residuals')
axis([0 3000 -.02 .02])
axis tight
