function [A,B,C,D,Q,R,X,P,bestLL,outLog]=randomStartEM(Y,U,nd,opts)

%First iter:
fprintf(['\n Starting rep 0 (fast one)... \n']);
%Pre-process optional flags:
if isa(U,'cell')
  Nu=size(U{1},1);
else
  Nu=size(U,1);
end
opts=processEMopts(opts,Nu);
outLog=struct();
opt1=opts;
opt1.fastFlag=true; %Enforcing fast filtering
opt1.Niter=max([opt1.Niter,500]); %Very fast evaluation of initial case, just to get a benchmark.
warning('off','EM:logLdrop') %If samples are NaN, fast filtering may make the log-L drop (smoothing is not exact, so the expectation step is not exact)
warning('off','EM:fastAndLoose')%Disabling the warning that NaN and fast may be happening
[A,B,C,D,Q,R,X,P,bestLL,startLog]=EM(Y,U,nd,opt1);
warning('on','EM:logLdrop')
warning('on','EM:fastAndLoose')
if opts.logFlag
    outLog.opts=opts;
    outLog.startLog=startLog;
    tic;
end
opts.targetLogL=bestLL;
for i=1:opts.Nreps
    fprintf(['\n Starting rep ' num2str(i) '. Best logL so far=' num2str(bestLL,8) '... \n']);

    %Initialize starting point:
    Xguess=guess(nd,Y,U,opts);

    %Optimize:
    [Ai,Bi,Ci,Di,Qi,Ri,Xi,Pi,logl,repLog]=EM(Y,U,Xguess,opts);

    %If solution improved, save and display:
      if logl>bestLL
          A=Ai; B=Bi; C=Ci; D=Di; Q=Qi; R=Ri; X=Xi; P=Pi;
          bestLL=logl;            opts.targetLogL=bestLL;
          disp(['Success, best logL=' num2str(bestLL,8)])
      end
      if opts.logFlag
          outLog.repLog{i}=repLog;  outLog.repRunTime(i)=toc;   tic;
      end
end

disp(['Refining solution...']);
opts.Niter=1e4;
opts.convergenceTol=1e-9;
opts.targetTol=0;
opts.fastFlag=false; %Patience. Disabling fast mode may not be needed for accurate results if there is no missing data. Further, if the system has very slow poles, then fast mode set simply to true may not increase speed anyway (but forcing some number larger than 1 will result in speed, but there is a trade-off with accuracy).
[Ai,Bi,Ci,Di,Qi,Ri,Xi,Pi,bestLL1,refineLog]=EM(Y,U,X,opts,P); %Refine solution, sometimes works
if bestLL1>bestLL
    A=Ai; B=Bi; C=Ci; D=Di; Q=Qi; R=Ri; X=Xi; P=Pi; bestLL=bestLL1;
end

disp(['End. Best logL=' num2str(bestLL,8)]);
if opts.logFlag
    outLog.repfineLog=refineLog;
    outLog.refineRunTime=toc;
end
end

function Xguess=guess(nd,Y,U,opts)
    if isa(U,'cell')
        u=cell2mat(U);
        y=cell2mat(Y);
    else
        y=Y;
        u=U;
    end
    [ny,N]=size(y);
    A1=diag(exp(-1./exp(log(N)*rand(nd,1)))); %WLOG, diagonal matrix with log-uniformly spaced time-constants in the [1,N] interval
    %I think the sign above is unnecessary
    B1=ones(nd,size(u,1)); %WLOG
    Q1=(abs(randn)+1e-4)*eye(nd); %Needs to be psd
    C1=randn(ny,nd)/ny; %WLOG
    D1=randn(ny,size(u,1));
    [~,Xsmooth]=fwdSim(u,A1,B1,zeros(1,nd),zeros(1,size(u,1)),[],Q1,[]);
    z=y-C1*Xsmooth(:,1:end-1)-D1*u;
    idx=~any(isnan(z));
    z=z(:,idx);
    R1=z*z'/size(z,2) + C1*Q1*C1'; %Reasonable estimate of R
    %[Xguess]=statKalmanSmoother(y,A1,C1,Q1,R1,[],[],B1,D1,u,opts);
    %Alternative:
    [Xguess]=statInfoSmoother2(y,A1,C1,Q1,R1,[],[],B1,D1,u,opts);
    Xguess=medfilt1(Xguess,9,[],2); %Some smoothing to avoid starting with very ugly estimates
    if isa(U,'cell')
        Xguess=mat2cell(Xguess,size(Xguess,1),cellfun(@(x) size(x,2),Y));
    end
end

function Xguess=guess2(nd,Y,U,opts)
    if isa(U,'cell')
        u=cell2mat(U);
        y=cell2mat(Y);
    else
        y=Y;
        u=U;
    end
    [ny,N]=size(y);
    C1=randn(ny,nd)/ny;
    D1=randn(ny,size(u,1));
    Xguess=C1\(y-D1*u);
    Xguess=medfilt1(Xguess,9,[],2); %Some smoothing to avoid starting with very ugly estimates
    if isa(U,'cell')
        Xguess=mat2cell(Xguess,size(Xguess,1),cellfun(@(x) size(x,2),Y));
    end
end
