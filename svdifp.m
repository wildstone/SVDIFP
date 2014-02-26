function [S,V,res] = svdifp(varargin) 
%
%   SVDIFP: Find a few smallest or largest singular values of an M-by-N matrix A
%       with M >= N. If M < N, the transpose of A will be used for computation.
%
%   [S, V] = svdifp(A) returns the smallest singular value S and
%           corresponding right singular vector V of A.
%
%   [S, V] = svdifp(A,K) returns the K smallest singular values S and
%           corresponding right singular vectors V of A.
%
%   [S, V] = svdifp(A,K,'L') returns the K largest singular values and
%           corresponding right singular vector of A.
%
%   [S, V, res] = svdifp(...) returns residuals for corresponding singular
%           pairs defined by res = norm(A' * A * v - sigma^2 * v).
%
%   svdifp(A,opt), svdifp(A,K,opt), svdifp(A,K,'L',opt)
%   take the following optional inputs defined by opt:
%     opt.INITIALVEC:
%       a matrix whose i-th column is the i-th initial approximate right singular
%       vector.
%     opt.COLAMD:
%       set to 0 to diable column approximate minimum degree permutation;
%       default: 1.
%     opt.TOLERANCE:
%       termination tolerance for the 2-norm of residual;
%       default: 10*eps*sqrt(n)*||A||^2.
%     opt.MAXIT:
%       set the maximum number of (outer) iterations; default: 1000.
%     opt.INNERIT:
%       set a fixed inner iteration to control the memory requirement;
%       default: between 1 and 128 as adaptively determined.
%     opt.USEPRECON:
%       set to 0 to disable preconditioning; default: 1.
%     opt.SHIFT:
%       set shift to  an approximated singular value; default: 0.
%     opt.ADSHIFT: 
%       set adshift to 1 to choose shift adaptively;
%       default: 0 if computing the smallest singular value and 1 if
%       computing the largest singular value.
%     opt.RIFTHRESH:   
%       a threshold between 0 and 1 used in RIF for computing
%           preconditioner; default: 1e-3;
%           If set to 0, the preconditioner is constructed by the QR
%           factorization.
%     opt.ZRIFTHRESH:
%       a threshold between 0 and 1 used for dropping the Z-factor in RIF;
%           default: 10*eps*sqrt(n)*||A||^2.
%     opt.RIFNNZ:
%       a number between 1 and n which preallocates the nonzeros in
%           each column of Z in RIF. default: 1000
%     opt.DISP:
%       set to 0 to disable on-screen display of output, and to any other
%       numerical value to enable display; Default: 1.
%   
%  SVDIFP computes the singular values of A through computing the eigenvalues
%  of A'A using the inverse free preconditioned Krylov subspace method
%  (EIGIFP). The basic SVDIFP algorithm is described in
%      
%       Q. Liang, Q. Ye, Computing Singular Values of Large Matrices
%       With Inverse Free Preconditioned Krylov Subspace Method, submitted
%
%  SVDIFP uses preconditioning constructed by the robust incomplete factorization (RIF)
%  (see M. Benzi and M. Tuma. A Robust Preconditioner with Low Memory Requirements for Large
%  Sparse Least Squares Problems, SIAM J. Scientific Computing, 25 (2003), 499-512).
%  Two routines of RIF (RIF.c or RIFLDL.m) are provided with RIF.c being the
%  more efficient choice. To use RIF.c, invoke MATLAB's mex functionality by
%       mex -largeArrayDims RIF.c 
%  Our RIF routines are based on the implementation of rifnrimex.F by Benzi
%  and Tuma available at http://www2.cs.cas.cz/~tuma/sparslab.html.
%
%  SVDIFP is developed by Qiao Liang (qiao.liang@uky.edu) and Qiang Ye
%  (qiang.ye@uky.edu). This work was supported by NSF under Grant DMS-1317424.
%
%  This program is provided for research and educational use and is
%  distributed through http://www.ms.uky.edu/~qye/software.
%  Neither redistribution nor commercial use is permitted without
%  consent of the authors.
%
%  This version dated January 7 2014



% process inputs
A = varargin{1};

if ~issparse(A) 
	A = sparse(A);
	fprintf(1, 'Warning: A is not in the sparse format.\n');             
end  

[m,n] = size(A);
if m < n
    A = A';
    [m,n] = size(A);
    fprintf(1, 'Warning: m < n, left singular vector will be returned!\n');
end




% Some default values
k = [];
maxsym = [];
FindMax = 0;
outputYes = 1;
shift = [];
rifthresh = 0.001;
iterMax = 1000;
tolerance = [];
X = [];
adaptiveshift = 0;
rifnnz = 1000;
zrifthresh = [];
usePrecon = 1;
mValue = 0;
iscolamd = 1;

% Check Inputs: 
if nargin > 1 
    if ~isstruct(varargin{2})
        k = varargin{2};
        if ~isnumeric(k) || ~isscalar(k) || ~isreal(k) || (k>n) || ...
                (k<0) || ~isfinite(k)
            error('SVDIFP: Number of singulars requested, k, must be a positive integer <= n');
        end
        if round(k) ~= k
           fprintf(1, 'Warning: Number of singulars requested, k, must be a positive integer <= n. Rounded number used\n'); 
           k = round(k);
        end
    end
    if (nargin > 2) && (~isstruct(varargin{3}))
        maxsym = varargin{3};
        FindMax = 1;
        if ~strcmp(maxsym,'L')
            error('SVDIFP: Third argument must be the string ''L''.')
        end
    end
end

% Get options
if nargin > 1 + ~isempty(k) + ~isempty(maxsym)
    options = varargin{2 + ~isempty(k) + ~isempty(maxsym):nargin};
    if ~isstruct(options)
        error('SVDIFP: Input Error: too many input parameters.'); 
    end
    
    names = fieldnames(options);   

    % Check whether initial approximated right singular vectors are given
    I = strcmpi('INITIALVEC',names);
    if any(I)
        X = options.(names{I});
        if size(X,1) ~= n
            if size(X,1) == m
                X = A' * X;
                fprintf(1, 'Warning: m<n, provided initial vector will be changed to be A''*X!\n');
            else
                error('SVDIFP: Input Error: incorrect size of initial vectors');
            end
        end
        if size(X,2) ~= k
           fprintf(1,'Warning: Number of initial vectors not equal to k\n');
        end    
    end
    
    % Check whether COLAMD is disabled
    I = strcmpi('COLAMD',names);
    if any(I)
        iscolamd = options.(names{I});
    end
    

    % Check whether tolerance of eigifp is given
    I = strcmpi('TOL',names);
    if any(I)
       tolerance = options.(names{I});
       if(tolerance <= 0)
           error('SVDIFP: Input Error: Invalid tolerance input.');
       end   
    end

    % Check whether maxItertion of svdifp is given
    I = strcmpi('MAXIT',names);
    if any(I)
        iterMax = options.(names{I});
    end
    
    % Check whether  mValue is given
    I = strcmpi('INNERIT',names);
    if any(I)
        mValue = options.(names{I});
    end
        
    % Check whether preconditioning is disabled        
    I = strcmpi('USEPRECON',names);
	if any(I)
        usePrecon = options.(names{I});
	end


    % Check whether approximated singular value is given
    I = strcmpi('SHIFT',names);
    if any(I)
       shift = options.(names{I});
    end
    
	% Check whether shift is determined by adaptive method
    I = strcmpi('ADSHIFT', names);
    if any(I) 
        adaptiveshift = options.(names{I});        
        if ~isempty(shift) && adaptiveshift ~= 0
           shift = [];
           fprintf(1,'Warning: Choosing shift adaptively, shift provided will not be used!\n'); 
        end        
    end  


    % Check whether dropping thershold of L in RIF is given
    I = strcmpi('RIFTHRESH',names);
    if any(I)
        rifthresh = options.(names{I});
    end
    
    % Check whether dropping threshold of Z in RIF is given
    I = strcmpi('ZRIFTHRESH',names);
    if any(I)
        zrifthresh = options.(names{I});
    end
    
    % Check whether rifnnz of RIF is given
    I = strcmpi('RIFNNZ',names);
    if any(I)
        rifnnz = options.(names{I});
    end
    
    % Check whether display option is off
    I = strcmpi('DISP',names);
    if any(I)
       outputYes = options.(names{I});
    end
end

% Check whether COLAMD is disabled
PM = eye(n,n);
if iscolamd    
    p = colamd(A);
    A = A(:,p);
    PM = PM(:,p);
else 
    fprintf('COLAMD is disabled\n');
end


% Check whether k is given
if isempty(k)
    k = 1;
end

% Check whether shift is given
if isempty(shift) && adaptiveshift == 0
   shift = 0; 
end


% estimated norm of A'A
normata = normest(A) ^ 2; 

% Check whether dropping threshold is given, if not, set it to be default
% value
tolerance0 = 10 * eps * normata * sqrt(n);
if isempty(tolerance) || tolerance < tolerance0 
    tolerance = tolerance0;
end

% Check whether zrifthresh is given, if not, set it to be default value
if isempty(zrifthresh)
   zrifthresh = tolerance0; 
end


% Main function
[S, V,res] = ifreesvd(A, FindMax, n,mValue,shift,rifthresh,zrifthresh,rifnnz,...
    tolerance,iterMax,k,X,normata,outputYes,usePrecon);
V = PM * V;


   
    
    
   
function [sv, x, r,  diff, rDiff, lam_max, res] = ...
     parnob(C,FindMax, L, sv, x, r,  diff, rDiff,  m, tol, SVecs,  lambda_1, lambda_max)
%  
%   generate othonormal basis V of preconditioned (by L) Krylov subspace
%   by m steps of Arnoldi algorithm and then compute the Ritz value  
%   and Ritz vectors 
%   
%   x: initial approximated right singular vector
%   SVecs =  converged singular vectors
%

% initialization and normalization 
n = size(x,1);   
nr = size(C,1);

% deflation for converged singular vectors
if nargin == 13
    count = size(SVecs,2);
    if FindMax == 1
        % larger singular values are shifted to 0
        K = speye(count);
        K = -K;
        C = C + C * SVecs * K * SVecs';
    else
        % smaller singular values are shifted to sqrt(lambda_max)
        dims = count + nr;
        SIGMA = diag(lambda_max - lambda_1);
        SIGMA = sqrt(SIGMA);
        K = speye(dims);
        K1 = K(:,1:nr);
        K2 = K(:,nr+1:end);
        K2 = K2 * SIGMA;
        C = K1 * C + K2 * SVecs';
        nr = dims;
    end
end



% CV = WBm
V = zeros(n,m);
W = zeros(nr,m);
Wr = V; 
Bm = zeros(m,m);

temp = x' * x;
temp = sqrt(temp);
V(:,1) = x / temp;

W(:,1) = C*V(:,1); 
Bm(1,1) = norm(W(:,1));
W(:,1) = W(:,1) / Bm(1,1);
    
r = r / temp; 
Wr(:,1) = r;

% Loop for Lanczos iteration and Gram-Schmidt
for i = 2:m-1                
    % Apply preconditioner if given 
    if ~isempty(L)
        if isnumeric(L)         
            r = L \ r;
            r = (L') \ r;    
        else 
            r = feval(L,r); 
        end
    end
    % generate new basis vector 
    for k = 1:(i-1)
        temp = V(:,k)'*r;
        r = r - temp * V(:,k);
    end
    % reorthogonalization if m > 6 
    if( m > 6)  
        for k = 1:(i-1)        
            temp = V(:,k)' * r;
            r = r - temp * V(:,k);
        end
    end
    
    if norm(r) == 0 
        m = i;
        break 
    end
    % normalize and save new basis vector 
    temp = norm(r);
    V(:,i) = r;
    V(:,i) = V(:,i) / temp;
    if (isa(C,'function_handle'))
        r = feval(C, V(:,i))-sv^2 * V(:,i);
    else
        W(:,i) = C * V(:,i);
        r = C' * W(:,i) - sv^2 * V(:,i);
    end
    
    % orthogonalize W(:,i)
    for k = 1:(i-1)
        Bm(k,i) = W(:,k)' * W(:,i);
        W(:,i) = W(:,i) - Bm(k,i) * W(:,k);
    end
    % reorthogonalization if m > 6 
    if m > 6  
        for k = 1:(i-1),        
            temp = W(:,k)' * W(:,i);
            W(:,i) = W(:,i) - temp * W(:,k);
        end
    end
    Bm(i,i) = norm(W(:,i));
    W(:,i) = W(:,i) / Bm(i,i);    
    Wr(:, i) = r;   
end 

% add the diff vector to the basis
% and complete the projection Bm 
diffNorm = sqrt(diff' * diff);
for k = 1:m-1
    temp = V(:,k)' * diff;
    diff = diff - temp * V(:,k);
    rDiff = rDiff - temp * Wr(:,k);
end

% reorthogonalization if m > 6
if m > 6
    for k = 1:m-1
        temp = V(:,k)' * diff;
        diff = diff - temp * V(:,k);
        rDiff = rDiff - temp * Wr(:,k);
    end
end 

temp = norm(diff);

% check and add diff only if it's significant
if temp <= 1e-8 * diffNorm || temp == 0 
        m = m-1;
elseif temp <= 1e-2*diffNorm
    % recompute (A-lambda)diff if necessary
    V(:,m) = diff / temp;
    diff = V(:,m); 
    if isa(C,'function_handle')
        rDiff = feval(C, V(:,m)) - sv^2 * diff;
    else
        W(:,m) = C*V(:,m);
        rDiff = C' * W(:,m) - sv^2 * diff;
    end    
    Wr(:,m) = rDiff;    
    
    % orthogonalize W(:,m)
    for k = 1:(m-1)
        Bm(k,m) = W(:,k)' * W(:,m);
        W(:,m) = W(:,m) - Bm(k,m) * W(:,k);
    end
    % reorthogonalization if m > 6 
    if m > 6 
        for k = 1:(m-1),        
            temp = W(:,k)' * W(:,m);
            W(:,m) = W(:,m) - temp * W(:,k);
        end
    end
    Bm(m,m) = norm(W(:,m));
    W(:,m) = W(:,m) / Bm(m,m);     
else     
    V(:,m) = diff / temp; 
    Wr(:,m) = rDiff / temp;
    r = Wr(:,m);

    % orthogonalize W(:,m)
    W(:,m) = C * V(:,m);
    for k = 1:(m-1)
        Bm(k,m) = W(:,k)' * W(:,m);
        W(:,m) = W(:,m) - Bm(k,m) * W(:,k);
    end
    % reorthogonalization if m > 6 
    if m > 6
        for k = 1:(m-1),        
            temp = W(:,k)' * W(:,m);
            W(:,m) = W(:,m) - temp * W(:,k);
        end
    end
    Bm(m,m) = norm(W(:,m));
    W(:,m) = W(:,m) / Bm(m,m);         
end

% compute Ritz value and vector of projection

[~,D,U] = svd(Bm(1:m,1:m));
[delta, Isv] = sort(diag(D));  

if FindMax == 0
    svidx = 1;
else
    svidx = m;
end
U(:,Isv(svidx)) = U(:,Isv(svidx)) / norm(U(:,Isv(svidx))); 
sv = D(Isv(svidx),Isv(svidx)); 
x = V(:,1:m) * U(:,Isv(svidx));
r = Wr(:,1:m) * U(:,Isv(svidx));
sigma = (x' * r) / (x' * x);
lambda = sv^2 + sigma;
r = r - sigma * x; 

% update new diff and related vectors
% for the next iteration   
U(1,Isv(svidx)) = -(U(2:m,Isv(svidx))' * U(2:m,Isv(svidx))) / U(1,Isv(svidx)); 
diff = V(:,1:m) * U(1:m,Isv(svidx));
rDiff = Wr(:,1:m) * U(1:m,Isv(svidx)); 
rDiff = rDiff - sigma * diff;
res = norm(r,2);
lam_max = lambda + delta(size(delta,1)).^2;%Upper bound     

if (res < tol) || (m>10)         % recompute r if necessary 
    if (isa(C,'function_handle'))
        r = feval(C, x);
    else
        r = C' * (C * x);
    end
    lambda = (x' * r) / (x' * x); 
    r=r-lambda*x;
    res=norm(r,2);  
end

% use a new diff if converged. 
if res < tol 
        diff = V(:,1:m) * U(:,Isv(svidx + 1 - 2 * FindMax));
end


 
function [SVals, SVecs,Sres] = ...
    ifreesvd(C, FindMax,n, m,shift,rifthresh,zrifthresh,rifnnz, ...
    tol, itermax, k, X,  normA, outputYes,usePrecon)
%
%  The main function that carries out the (outer) iteration. 
%    
%  To be consistent with the notation in the paper, here we use C to denote
%  the target matrix and A = C' * C.
%

if ~isempty(shift) && usePrecon && FindMax == 0
    if shift == 0 && rifthresh == 0
        if outputYes ~= 0
            fprintf('===================================================================\n');
            fprintf('Computing precondioner of A using qr, if it takes too long, try increasing opt.rifthresh\n');
            fprintf('====================================================================\n');
        end
        R = qr(C,0);
        for i = 1:n
            if R(i,i) < 1e-8
                R(i,i) = 1e-8;
                R(i,(i+1):end) = 0;
            end
        end
        L = @(x) (R \ (R' \ x));
        startL = 1;
        Lcomputed = 2;
    else
        if outputYes ~= 0
            fprintf('===================================================================\n');
            fprintf('Computing RIF of A, if it takes too long, try increasing opt.rifthresh or decreasing rifnnz\n');
        end
        
        RIFexist = exist('RIF','file');
        if RIFexist == 3
            tic;
            R = RIF(C,shift,rifthresh,zrifthresh,rifnnz);
            toc;
        else
            fprintf('Couldn''t find RIF.mexa32 or RIF.mexa64, Matlab implementation RIFLDL is used instead\n');
            R = RIFLDL(C,shift,rifthresh,zrifthresh);
        end
        L = @(x) R' \ (R \ x);
        if outputYes ~= 0
            fprintf('Done!\n');
            fprintf('====================================================================\n');
        end
        startL = 1;
        Lcomputed = 1;
    end
else
    startL = 0;
    Lcomputed = 0;
end


% initialization 
conv = ones(itermax,1);
rate = ones(10,1);
Lambda_1 = zeros(k,1);
Lambda_Max = zeros(k,1);
SVals = zeros(k,1);
SVecs = zeros(n,k);
Sres = zeros(k,1);


% set initial random vector if not given
if ~isempty(X)
    x = X(:,1);
    normX = norm(x);
    if normX == 0 || size(x,1) ~= n
        error('IFREESVD: Input Error: Invalid input of the initial vector.');
    end
else
    x = rand(n,1) - 0.5 * ones(n,1);
    normX = norm(x);
end 
x = x / normX;


% loop for computing k singular values
for l = 1:k
    
    % initialization for each singular value iteration
    diff = zeros(n,1);        
    rDiff = diff; 
    temp = x' * x; 
    r = C' * (C * x);
    lambda = (x' * r) / temp;
    sv = sqrt(lambda);    
    r = r - lambda * x;
    res = norm(r);
    conv(1) = res;
    
    initialConv = 0; 
    changeCounter = -itermax;
    mValue = m;
    if mValue < 1
        mValue = min(n-1,2);
        if startL == 1
            mValue = 1; 
        end
        changeCounter = -10;
    end
    priorM = mValue; 
    priorRate = -eps;
        
    if outputYes ~=0
        fprintf(1,'\nComputing  %d-th Singular value of A:\n',l);
        fprintf(1,'  Iteration\tSingular Value\tResidual(A''Ax-lambda*x)\n');
        fprintf(1,'  ======================================================\n');
        fprintf(1,'  %d\t\t%E\t%E\n',1,sv,res); 
    end
    
    % iteration for computing l-th singular value
    for iter = 2:itermax        
        % if converged, set parameters for next singular value
        if res < tol   
            lambda_max = normA; 
            initialConv = 1;  
            break;
        end
        projSize = mValue + 2;  
        % compute Ritz value and vector by Krylov projection   
        if l>1
          % with deflation for l > 1      
          if startL == 0 
              % without preconditioning
                    [sv, x, r,  diff, rDiff,  lambda_max, res] = ...
              parnob(C, FindMax, [], sv, x, r,  diff, rDiff, projSize, tol,...
                          SVecs(:,1:l-1),  Lambda_1(1:l-1), Lambda_Max(1:l-1,:));
     
          else
              % with preconditioning
                    [sv, x, r,  diff, rDiff,  lambda_max, res] = ...
              parnob(C, FindMax, L, sv, x, r,  diff, rDiff,  projSize, tol,...
                          SVecs(:,1:l-1),  Lambda_1(1:l-1), Lambda_Max(1:l-1,:));
                   
          end           
        else
          % no deflation if l=1. 
          if ( startL == 0 )
              % without preconditioning
                    [sv, x, r,  diff, rDiff,  lambda_max, res] = ...
              parnob(C, FindMax, [], sv, x, r,  diff, rDiff,  projSize, tol);
               
          else
              % with preconditioning
                    [sv, x, r,  diff, rDiff, lambda_max, res] = ...
              parnob(C, FindMax, L, sv, x, r,  diff, rDiff,  projSize, tol);
                   
          end           
        end
        lambda = sv^2; 
        conv(iter) = res;        
        if (outputYes ~= 0)
            fprintf(1,'  %d\t\t%E\t%E\n',iter,sv,res);
        end
        
        % update tolerance and check convergence
        if res <= tol
            break;
        end

        % check on convergence rate and update mValue
        changeCounter = changeCounter + 1;
        if changeCounter >= 19 
            rate = [rate(2:10); aveRate(conv, iter - changeCounter + 4, iter)];
            [mValue, priorM, priorRate, fixM] = updateM(rate,mValue,priorM,priorRate,n);
            changeCounter = (changeCounter + fixM * itermax) * (1 - fixM);  
        elseif changeCounter >= 10 
            rate = [rate(2:10); aveRate(conv, iter - changeCounter + 4, iter)];
        end
 

        if (res <=0.01 || iter >= itermax/5) && (startL == 0) && usePrecon == 1
            startL = 1;
            Lcomputed = 1;            
            shift = sv;
            if outputYes ~= 0
                fprintf('===================================================================\n');
                fprintf('Computing RIF of A, if it takes too long, try increaseing opt.rifthresh\n');
            end
            RIFexist = exist('RIF','file');
            if RIFexist == 3
                R = RIF(C,shift,rifthresh,zrifthresh,rifnnz);
            else
                fprintf('Couldn''t find RIF.mexa32 or RIF.mexa64, Matlab implementation RIFLDL is used instead\n');
                R = RIFLDL(C,shift,rifthresh,zrifthresh);
            end
            L = @(x) R' \ (R \ x);
            if outputYes ~= 0
                fprintf('Done!\n');
                fprintf('====================================================================\n');
                fprintf('Start preconditioned SVDIFP\n');
            end
        end
        
        if (((res >= sqrt(tol)) && iter >= itermax/2) || ((res >= tol * 1e2) && iter >=itermax-50))...
                && Lcomputed <= 1 && usePrecon == 1
            Lcomputed = 2;
            if outputYes ~= 0
                fprintf('===================================================================\n');
                fprintf('ReComputing RIF of matrix A, if it takes too long, try increasing opt.rifthresh\n');
            end
            shift = sv;
            if RIFexist == 3
                R = RIF(C,shift,rifthresh,zrifthresh,rifnnz);
            else
                fprintf('Couldn''t find RIF.mexa32 or RIF.mexa64, Matlab implementation RIFLDL is used instead\n');
                R = RIFLDL(C,shift,rifthresh,zrifthresh);
            end
            L = @(x) R' \ (R \ x);            
            if outputYes ~= 0
                fprintf('Done!\n');
                fprintf('====================================================================\n');
                fprintf('Start a new preconditioned SVDIFP\n');   
            end
        end
    end

    % store singular values and others
    SVals(l) = sv;
    Lambda_1(l) = lambda;
    Lambda_Max(l) = lambda_max;
    temp = sqrt(x' * x);
    SVecs(:,l) = x / temp; 
    Sres(l) = res;

    % warn if not converged                         
    if res >= tol
        fprintf(1,'\n');
        fprintf(1,'Warning: Singular Value %d not converged to the tolerance within max iteration\n',l);
        fprintf(1,'         Residual = %E , the set tolerance = %E \n',res, tol);
        fprintf(1,'         Try decreasing opt.rifthresh, increasing opt.rifnnz, opt.maxit, opt.innerit, or choose a better shift\n');
    end
    if outputYes ~= 0
        fprintf(1,'  ------------\n');
        fprintf(1,'  Singular Value %d converged. \n',l); 
        fprintf(1,'Residual = %.16E , the set tolerance = %E \n',res, tol);
    end
    
    %  set next initial vector  
    if l < size(X,2)
        x = X(:,l+1);
        normX = norm(x);
        if normX == 0
            fprintf(1,'Invalid input of initial vector.\n');
        end
        x = diff;
    elseif initialConv==1 
        x = rand(n,1)-0.5*ones(n,1);  
    else
        x = diff;
    end
    x = x - SVecs(:,1:l) * (SVecs(:,1:l)' * x);
    normX = norm(x);
    if normX == 0
        if l<k 
            error('IFREESVD: Fail to form an initial vector.');
        end 
        return;
    end
    x = x / normX;  
end



function L = RIFLDL(A,shift,rifthresh,zrifthresh)
% Robust Incomplete factorization of A'A


[~,n] = size(A); % get number of columns of A

nzmax = 100000000;


L = spalloc(n,n,nzmax);

threshold = zeros(n,1);
for j = 1:n
   threshold(j) = rifthresh * normest(A(:,j));
end



P = zeros(n,1);
Z = speye(n,n);

for j = 1:n
    PrevZ = Z;
    
    % Compute <Azj,Azj> - shift ^ 2 * <zj,zj>
    
    % The first column of PrevZ is current zj
    tmpZj = PrevZ(:,1);
    tmpZjnorm = normest(tmpZj);
    tmpZj(abs(tmpZj) < zrifthresh * tmpZjnorm) = 0;
    AZj = A * tmpZj; 
    P(j) = AZj' * AZj - shift * shift * PrevZ(:,1)' * PrevZ(:,1);
    
    L(j,j) = sqrt(abs(P(j)));
    if P(j) < 1e-8
        L(j,j) = 1e-8;
        continue;
    end    
    
    % Get entries in L
    tmpcol = A(:,j+1:n)' * AZj;
    tmpcol = tmpcol / sqrt(P(j));
    tmpidx = (abs(tmpcol) < threshold(j+1:n));
    tmpcol(tmpidx) = 0;
    L(j+1:n,j) = tmpcol;    
    tmpcoef = tmpcol / sqrt(P(j));
    
    % Update columns of Z   
    Z = PrevZ(:,2:end) - PrevZ(:,1) * tmpcoef';
end



function rate = aveRate(conv, k, iter)
%
% compute average linear convergence rate of conv over steps k+1 to iter 
%
rate = 0; 
if iter-k < 2 || k<1  
    return;
end

y = log10(conv((k+1):iter));
xAve = (iter-k+1) / 2; 
xyAve = ((1:iter-k)*y) / (iter-k)-xAve*sum(y) / (iter-k); 
xAve = ((iter-k) ^ 2 - 1) / 12;
rate = xyAve / xAve;

        

function [mValue, priorM, priorRate, fixM] = updateM(rate, mValue, priorM, priorRate,n)       
%
%  Adaptive update of mValue: inner iteration 
%
fixM = 0;
maxm = min(n-1, 128);

                    % update m when rate stagnates  
if (max(rate)-min(rate)) < 0.1*(-rate(10)) || min(rate) > 0 
    k=2;                
    % increase m by k times, 
     % use larger k if slower convergence 
	if (rate(10) > -0.001) && 8 * mValue <= maxm
        k = 8; 
	elseif rate(10) > -0.01 && 4 * mValue <= maxm
        k = 4;
	end
	% increase m by testing acceleration rate 
	incFlag = (rate(10) / priorRate) * (priorM / mValue); 
    if incFlag > 1.05 
        if 2*mValue > maxm 
            fixM = -1;
        else
            priorM = mValue; 
            priorRate = rate(10); 
            mValue = k * mValue;
            fixM = 1;
        end  
	elseif rate(10) > -0.001 && 2*mValue <= maxm
        mValue = k * mValue;
        fixM = 1;  
	elseif rate(10) > -0.01 && 2*mValue <= maxm
        mValue = k * mValue;
        fixM = 1;   
	elseif incFlag < 0.9
        mValue = priorM;
        fixM = -1;  
    end
end




