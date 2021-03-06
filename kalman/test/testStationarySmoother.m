%% Create model:
D1=2;
D2=100;
N=1000;
A=[.95,0;0,.99];
B=(eye(size(A))-A)*ones(size(A,1),1); %WLOG, arbitrary scaling
U=[zeros(300,1);ones(N,1);zeros(N/2,1)]'; %Step input and then removed
C=randn(D2,D1);
D=randn(D2,1);
Q=eye(D1)*.0005;
R=eye(D2)*.01;

%% Simulate
NN=size(U,2);
x0=zeros(D1,1);
[Y,X]=fwdSim(U,A,B,C,D,x0,Q,R);
[Y1,X1]=fwdSim(U,A,B,C,D,x0,[],[]); %Noiseless simulation, for comparison

%% Do kalman smoothing with true params
tic
opts.fastFlag=0;
[Xs,Ps,Pt,Xf,Pf,Xp,Pp]=statKalmanSmoother(Y,A,C,Q,R,[],[],B,D,U,opts); %Kalman smoother estimation of states, given the true parameters (this is the best possible estimation of states)
tf=toc;
%% Visualize results
figure
for i=1:2
    subplot(3,1,i)
    plot(Xs(i,:),'DisplayName','Smoothed')
    hold on
    plot(Xf(i,:),'DisplayName','Filtered')
    plot(X(i,:),'DisplayName','Actual')

    legend
    if i==1
        title(['This runtime= ' num2str(tf)]);
    end
end
subplot(3,1,3)
for i=1:2
     hold on
     set(gca,'ColorOrderIndex',1)
    plot(Xs(i,:)-X(i,1:end-1),'DisplayName','Smoothed')
    plot(Xf(i,:)-X(i,1:end-1),'DisplayName','Filtered')
        set(gca,'ColorOrderIndex',1)
        aux=sqrt(mean((X(i,1:end-1)-Xs(i,:)).^2));
    b1=bar(1900+400*i,aux,'BarWidth',100,'EdgeColor','None');
    text(1850+400*i,1.2*aux,num2str(aux,3),'Color',b1.FaceColor)
    aux=sqrt(mean((X(i,1:end-1)-Xf(i,:)).^2));
    b1=bar(2000+400*i,aux,'BarWidth',100,'EdgeColor','None');
    text(1950+400*i,1.4*aux,num2str(aux,3),'Color',b1.FaceColor)
    grid on
end
title('Residuals')
axis([0 3000 -.02 .02])
