function [X,P,Xp,Pp,rejSamples,logL]=statKalmanFilter2(Y,A,C,Q,R,varargin)
%statKalmanFilter implements a Kalman filter assuming
%stationary (fixed) noise matrices and system dynamics
%The model is: x[k+1]=A*x[k]+b+v[k], v~N(0,Q)
%y[k]=C*x[k]+d+w[k], w~N(0,R)
%And X[0] ~ N(x0,P0) -> Notice that this is different from other
%implementations, where P0 is taken to be cov(x[0|-1]) so x[0]~N(x0,A*P0*A'+Q)
%See for example Ghahramani and Hinton 1996
%Fast implementation by assuming that filter's steady-state is reached after 20 steps
%INPUTS:
%
%OUTPUTS:
%
%See also: statKalmanSmoother, statKalmanFilterConstrained, KFupdate, KFpredict

% For the filter to be well-defined, it is necessary that the quantity w=C'*inv(R+C*P*C')*y
% be well defined, for all observations y with some reasonable definition of inv().
% Naturally, this is the case if R+C*P*C' is invertible at each step. In
% turn, this is always the case if R is invertible, as P is positive semidef.
% There may exist situations where w is well defined even if R+C*P*C' is
% not invertible (which implies R is non inv). %This requires both: the
% projection of y and of the columns of C onto the 'uninvertible' subspace
% to be  always 0. In that case the output space can be 'compressed' to a
% smaller dimensional one by eliminating nuisance dimensions. This can be done because
% neither the state projects onto those dims, nor the observations fall in it.
% Such a reduction of the output space can be done for efficiency even if
% the projection of y is non-zero, provided that R is invertible and the
% structure of R decouples those dimensions from the rest (i.e. the
% observations along those dims are uncorrelated to the dims corresponding
% to the span of C). Naturally, this is a very special case, but there are
% some easy-to-test sufficient conditions: if R is diagonal, positive, and
% rank(C)<dim(R), compression is always possible.


[D2,N]=size(Y); D1=size(A,1);
%Init missing params:
[x0,P0,B,D,U,opts]=processKalmanOpts(D1,N,varargin);
M=processFastFlag(opts.fastFlag,A,N);
if M~=N && any(isnan(Y(:)))
    warning('statKFfast:NaN','Requested fast filtering but data contains NaNs. No steady-state can be found for filtering. Filtering will not be exact. Proceed at your own risk.')
    %opts.fastFlag=false;
    %allows for fast-filtering anyway. The steady-state of K (but not P) may exist
    %under some special circumstances (e.g. missing data forms a regular
    %pattern, such as every other datapoint), and even if not, fast
    %filtering may still be good enough and thus acceptable. Issue warning
    %to make subject aware.
end

%TODO: Special case: deterministic system, no filtering needed. This can also be
%the case if Q << C'*R*C, and the system is stable

%Size checks:
%TODO

if nargout>6 %This is here to make sure that no code is using the legacy call with 7 outputs
    error('Too many outputs')
end

%Init arrays:
if isa(Y,'gpuArray') %For code to work on gpu
    Xp=nan(D1,N+1,'gpuArray');      X=nan(D1,N,'gpuArray');
    Pp=nan(D1,D1,N+1,'gpuArray');   P=nan(D1,D1,N,'gpuArray');
    rejSamples=false(D2,N,'gpuArray');
else
    Xp=nan(D1,N+1);      X=nan(D1,N);
    Pp=nan(D1,D1,N+1);   P=nan(D1,D1,N);
    rejSamples=false(D2,N);
end

%Priors:
prevX=x0; prevP=P0; Xp(:,1)=x0; Pp(:,:,1)=P0;

%Re-define observations to account for input effect:
Y_D=Y-D*U; BU=B*U;

%Check if any output dimensions have 0 or Infinite variances:
infVar=isinf(diag(R));
if any(infVar) %Any infinite variance is equivalent to ignoring the corresponding variables: it does not matter for the filtering itself.
    %However, it will lead to -Inf log-likelihood (adds a -Inf offset associated with each infinite eigenvalue of R).
    %Thus, I remove the relevant components to get a non-infinite log-likelihood.
    warning('statKF:infObsVar','Provided model has infinite variance for some observation components. Ignoring those components for log-likelihood calculation purposes.');
    Y_D=Y_D(~infVar,:);
    R=R(~infVar,~infVar);
    %D=D(~infVar,:);
    C=C(~infVar,:);
end
[~,dR]=ldl(R);
zeroVar=(diag(dR)==0);
if any(zeroVar) %0 variance is very problematic: it means that the output equation is at least partially deterministic, which doesn't bode well with the stochastic framework
    error('statKF:zeroObsVar','Provided model has 0 observation variance for some dimensions, this is incompatible with the Kalman framework (and unlikely in reality!). If uncertainties are truly 0, try reducing the model by separating the deterministic and stochastic components');
    %To do: offer a function to decouple deterministic and stochastic
    %observation dimensions, and run appropriate state estimation
    %algorithms on each of the components (where naturally the stochastic
    %part will be equivalent to the kalman filter projected onto the states
    %known exactly from the deterministic part)
end

%Define constants for sample rejection:
logL=nan(1,N); %Row vector
rejThreshold=0;
if opts.outlierFlag
  rejThreshold=chi2inv(.99,D2);
end

%Reduce model if convenient for efficiency:
if D2>D1 && ~opts.noReduceFlag %Reducing dimension of problem for speed
    [CtRinvC,~,CtRinvY,~,logLmargin]=reduceModel(C,R,Y_D);
    C=CtRinvC; R=CtRinvC; Y_D=CtRinvY;  D2=D1; rejSamples=rejSamples(1:D1,:);
    S=sqrtm(inv(R));
    R=eye(size(R)); C=eye(size(R)); Y_D=S*Y_D; A=S*CtRinvC*A/(S*CtRinvC); BU=S*CtRinvC*BU; Q=(S*CtRinvC)*Q*(S*CtRinvC)';
else
    logLmargin=0;
end

%For the first steps do an information update if P0 contains infinite elements
firstInd=1;
infVariances=isinf(diag(prevP));
if any(infVariances) %At least one initial variance was infinite
    %Run info filter until D1 non-nan samples have been processed: this is enough to resolve all uncertainties down from infinity if the system is observable
    %Strictly speaking, we should only need to process N = G - D2 + 1 ;
    %non-NaN samples, where G is the number of infinity uncertainties, D2 is the
    %dimension of the observation. This is true for an observable system,
    %otherwise we would need to find the observable/unobservable
    %decomposition and use the number of dimensions corresponding to the
    %observable part.
    Nsamp=find(cumsum(~any(isnan(Y_D),1))>=D1,1,'first');
    [ii,I,ip,Ip]=statInfoFilter2(Y_D(:,1:Nsamp),A,C,Q,R,prevX,prevP,B,zeros(D2,size(U,1)),U(:,1:Nsamp),opts);
    for kk=1:Nsamp
        logL(kk)=-Inf; %Warning: if variance was inifinte, then logL(firstInd)=-Inf!
        %Transform results from information to state estimates:
        [X(:,kk),P(:,:,kk)]=info2state(ii(:,kk),I(:,:,kk));
        [prevX,prevP]=info2state(ip(:,kk+1),Ip(:,:,kk+1));
        Xp(:,kk+1)=prevX;    Pp(:,:,kk+1)= prevP;
    end
    firstInd=Nsamp+1;
end

%Run filter for remaining steps:
i=firstInd;
processedSamples=0;
[prevU,prevD]=eig(prevP,'vector');
d=sqrt(prevD);
UD=prevU.*sqrt(d)';
I=eye(size(prevP));
%prevU=single(prevU);
%prevD=single(prevD);
%Y_D=single(Y_D);
%prevX=single(prevX);
%A=single(A);
%Q=single(Q);
%BU=single(BU);
while i<=N && processedSamples<M
%for i=firstInd:M
  y=Y_D(:,i); %Output at this step

  %First, do the update given the output at this step:
  if ~any(isnan(y)) %If measurement is NaN, skip update.
      prevD=prevD./(prevD+1);
      %Uinnov=prevU'*(y-prevX);
      d=sqrt(prevD);
      UD=prevU.*d';
      prevP=UD*UD';
      prevX=prevX+prevP*(y-prevX);
      %UD=prevU.*prevD';
      %prevP=UD*prevU;
      %prevX=prevX+UD*Uinnov;
      logL=NaN; %To do
        icS=NaN; %To do
     processedSamples=processedSamples+1;
  end
  X(:,i)=prevX;  P(:,:,i)=prevP; %Storing old P to avoid computing it

  %Then, predict next step:
  prevX=A*prevX+BU(:,i);
  %prevP=A*prevP*A'+Q; %Should ensure symmetricity?
  AP=A*UD;
  prevP=AP*AP'+Q;
  %[prevU,prevD]=eig(prevP,'nobalance','vector');
  [prevU,prevD]=svd(prevP);
  prevD=diag(prevD);
  %To do: eliminate less than 0 eigenvalues.
  %[prevU,prevD]=eig(prevP,I,'chol','vector');
  if nargout>2 %Store Xp, Pp if requested:
      Xp(:,i+1)=prevX;   Pp(:,:,i+1)=prevP;
  end
  i=i+1;
end

if i<=N %Do the fast filtering for any remaining steps:
%(from here on, we assume stady-state behavior to improve speed).
    %Steady-state matrices:
    prevX=X(:,i); Psteady=P(:,:,i); %Steady-state UPDATED state and uncertainty matrix
    PpSteady=prevP;
    CicS=C'*icS;
    Ksteady=PpSteady*C'*(icS*icS');
    Gsteady=eye(size(prevP))-PpSteady*(CicS*CicS');% %I-K*C,

    %Pre-compute matrices to reduce computing time:
    GBU_KY=Gsteady*BU(:,i:N-1)+Ksteady*Y_D(:,i+1:N); %The off-ordering is because we are doing predict (which depends on U(:,i-1)) and update (which depends on Y(:,i)
    GA=Gsteady*A;

    %Assign all UPDATED uncertainty matrices:
    P(:,:,i+1:end)=repmat(Psteady,1,1,N-i);

    %Check that outlier flag is disabled and that no data is missing
    if opts.outlierFlag %|| any(isnan(GBU_KY(:)))%Should never happen in fast mode
       error('KFfilter:outlierRejectFast','Outlier rejection is incompatible with fast mode.')
    end

    %Loop for remaining steps to compute x:
    for j=i+1:N
        gbu_ky=GBU_KY(:,j-i);
        if ~isnan(gbu_ky) %Have actual observation (not NaN)
            prevX=GA*prevX+gbu_ky; %Predict+Update, in that order.
        else %Observation is missing or rejected, cannot do update
            %Warning: if this happens, then you shouldn't be using fast
            %mode, as the Kalman filter does not reach a steady-state when
            %data is missing. This is, at best, a heuristic calculation.
            %(the value of Psteady defined above is not really
            %steady-state, and Ksteady may or may not be depending on
            %whether the missing data is missing in a regular pattern).
            %Consistently, the value of Pp defined below cannot apply to
            %all samples.
            prevX=A*prevX+BU(:,j); %Predict only;
        end
        X(:,j)=prevX;
    end
    if nargout>2 %Compute Xp, Pp only if requested:
        Xp(:,2:end)=A*X+B*U; Pp(:,:,i+2:end)=repmat(A*Psteady*A'+Q,1,1,size(Y,2)-i);
        if nargout>4; Innov=Y_D-C*Xp(:,1:end-1);  logL(i+1:end)=logLnormal(Innov(:,i+1:end),[],icS');
        end
    end
end

%Compute mean log-L over samples and dimensions of the output:
if firstInd~=1
    %warning('statKF:logLnoPrior',['Filter was computed from an improper uniform prior as starting point. Ignoring ' num2str(firstInd-1) ' points for computation of log-likelihood.'])
end
aux=logL+logLmargin;
logL=nansum(aux(firstInd:end)); %Full log-L
%logL=nanmean(aux(firstInd:end))/size(Y,1); %Per-sample, per-dimension of output PROVIDED (not necessarily used)

%TO do:
%Change back the states and covariances to the desired reference frame
%Note: for EM this is unecessary, as EM only requires the states in SOME
%reference frame, not in any particular one.
noUndoFlag=false;
if ~noUndoFlag
    X=S*X;
    Xp=S*Xp;
    iS=inv(S);
   for i=1:N
       P(:,:,i)=iS*P(:,:,i)*iS'; %PSD can be enforced by storing U and reconstructing P in a sqrt way
       Pp(:,:,i)=iS*Pp(:,:,i)*iS';
   end
end
end
