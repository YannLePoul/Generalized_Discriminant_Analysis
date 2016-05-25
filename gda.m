function mappedData = gda(data,trainData,trainLabel,nDim,options)

% GDA Performs Generalized Discriminant Analysis, a non-linear feature
% dimensionality reduction technique.
% 
% GDA is one of dimensionality reduction techniques, which projects a data 
% matrix from a high-dimensional space into a low-dimensional space by 
% maximizing the ratio of between-class scatter to within-class scatter. 
% 
% 
% Inputs:
%       data:           p-dimensional matrix containing the high-dimensional data to be projected
%                       p:  number of dimensions in high-dimensional space
%
%       trainData:      pxn matrix containing the high-dimensional training data
%                       n:  number of training samples
% 
%       trainLabel:     Row vector of length n containing the class labels for training data
% 
%       nDim:           Numer of dimensions to be retained (nDim < c)
%                       Default:    c-1
%                       c:  number of classes
% 
%       options:        Please see the kernel function (kernel.m).
%         
% 
% Output:
%       mappedData:      nDim-dimensional projected data matrix
% 
% 
% Sample use:
% trainGda  =  gda(trainData,trainData,trainLabel);     % Project the training data matrix into a low-dimensional space
% testGda  =  gda(testData,trainData,trainLabel);       % Project the test data matrix into a low-dimensional space
% 
% 
% 
%   Details can be found in Section 4.3 of:
%   
%   M. Haghighat, S. Zonouz, M. Abdel-Mottaleb, "CloudID: Trustworthy 
%   cloud-based and cross-enterprise biometric identification," 
%   Expert Systems with Applications, vol. 42, no. 21, pp. 7905-7916, 2015.
% 
% 
% 
% (C)	Mohammad Haghighat, University of Miami
%       haghighat@ieee.org
%       PLEASE CITE THE ABOVE PAPER IF YOU USE THIS CODE.
% 
% Thanks to Dr. Saeed Meshgini for his helps.



if(size(data,1) ~= size(trainData,1))
   error('DATA and TRAINDATA must be in the same space with equal dimensions.');
end

if(size(trainData,2) ~= size(trainLabel,2))
   error('The length of the TRAINLABEL must be equal to the number of columns in TRAINDATA.');
end

if (~exist('options','var'))
   options.KernelType='linear';
end


% Separate samples of each class in a cell array

[u, ~, idxU] = unique(trainLabel);
c = numel(u);
dataCell = cell(1,c);
nSample = zeros(1,c);
for i = 1:c
    ind = find(idxU==i);
    nSample(i) = numel(ind);
    dataCell{1,i} = trainData(:,ind);
end
clear trainLabel


% Create class-specific kernel for the training data

kTrainCell = cell(c,c);
for p = 1:c
    for q = 1:c
%% test if the results are similar with vectorized version of the code
%         classP = rand(3,30);
%         classQ = rand(3,30)+3;
%        Kpq = zeros(size(classP,2),size(classQ,2));
%        options.KernelType = 'wave';
%         for i = 1:size(classP,2)
%             for j = 1:size(classQ,2)
%                 Kpq(i,j) = kernel(classP(:,i),classQ(:,j),options);
%             end
%         end
%         Kpq2 = f_kernel(classP,classQ,options);
%         disp(isequal(Kpq,Kpq2))
        %%
        Kpq = f_kernel(dataCell{1,p},dataCell{1,q},options);
        kTrainCell{p,q} = Kpq;
    end
end
kTrain = cell2mat(kTrainCell);
clear kTrainCell 


% Make data have zero mean

[~,n] = size(trainData);
One = (1/n) * ones(n,n);
zeroMeanKtrain = kTrain - One*kTrain - kTrain*One+One*kTrain*One;
clear trainData


% Create the block-diagonl W matrix

wTrainCell=cell(c,c);
for p = 1:c
    for q = 1:c
        if p == q
            wTrainCell{p,q}=(1/nSample(p))*ones(nSample(p),nSample(q));
        else
            wTrainCell{p,q}=zeros(nSample(p),nSample(q));
        end
    end
end
wTrain = cell2mat(wTrainCell);
clear wTrainCell


% Decompose zeroMeanKtrain using eigen-decomposition

[P, gamma] = eig(zeroMeanKtrain);
diagonal = diag(gamma);
[~,index] = sort(diagonal,'descend');
gamma = diagonal(index);
P = P(:,index);


% Remove eigenvalues with relatively small value

maxEigVal = max(abs(gamma));
zeroEigIndex = find((abs(gamma)/maxEigVal)<1e-6);
gamma(zeroEigIndex) = [];
P(:,zeroEigIndex) = [];


% Normalize eigenvectors

nEig = length(gamma);
for i = 1:nEig
    P(:,i) = P(:,i)/norm(P(:,i));
end


% Compute eigenvectors (beta) and eigenvalues (lambda)

BB = (P')*(wTrain)*(P);
[beta, lambda] = eig(BB);
diagonal = diag(lambda);
[~, index] = sort(diagonal,'descend');
lambda = diagonal(index);
beta = beta(:,index);
clear BB wTrain


% Remove eigenvalues with relatively small value

maxEigVal = max(abs(lambda));
zeroEigIndex = find((abs(lambda)/maxEigVal)<1e-6);
lambda(zeroEigIndex) = [];
beta(:,zeroEigIndex) = [];


% Compute eigenvectors (alpha) and normalize them

gamma = diag(gamma);
alpha = (P/gamma)*beta;
nEig = length(lambda);
for i = 1:nEig
    scalar = sqrt((alpha(:,i).')*(zeroMeanKtrain)*(alpha(:,i)));
    alpha(:,i)=alpha(:,i)/scalar;
end
clear zeroMeanKtrain P gamma beta


% Dimensionality reduction (if nDim is not given, nEig dimensions are retained):

if (~exist('nDim','var'))
   nDim = nEig;       % nEig is the maximum possible value (the rank of subspace)
elseif (nDim > nEig)
   warning(['Target dimensionality reduced to ' num2str(nEig) '.']);
end

w = alpha(:,1:nDim);    % Projection matrix


% Create class-specific kernel for all data points:

[~,nPrime] = size(data);
kDataCell = cell(c,1);
for p = 1:c
    kDataCell{p,1} = f_kernel(dataCell{1,p},data,options);
end
kData = cell2mat(kDataCell);
clear data dataCell kDataCell


% Make data zero mean

Oneprime = (1/n)*ones(n,nPrime);
zeroMeanKdata = kData - kTrain*Oneprime - One*kData+One*kTrain*Oneprime;
clear kTrain kData


% Project all data points non-linearly onto a new lower-dimensional subspace (w):

mappedData = (w.') * (zeroMeanKdata);

end



function k = f_kernel(u,v,options)

% KERNEL determines kernel function for kernel-based machine learning 
% methods including Support Vector Machines.
% 
% 
% 
% Inputs:
%           u:          First input vector (p dimensional column vector)
%           v:          Second input vector (p dimensional column vector)
%           options:    Struct value in Matlab
%                       The fields in options that can be set:
% 
%                       options.KernelType:	choices are:
% 
%                           'linear'        without kernel function (no parameter)(default value)
%                           'poly'          simple polynomial kernel function (with 1 parameter which is degree of polynomial)
%                           'polyplus'      polynomial kernel function (with 1 parameter which is degree of polynomial)
%                           'sphnorm_poly'  spherically normalized polynomial kernel function (with 2 parameters: 
%                                           the first one is degree of polynomial & the second one is spherical normalization parameter)
%                           'rbf'           Gaussian (RBF) kernel function (with 1 parameyer which is width of RBF (sigma)
%                           'wave'          wavelet kernel function (with 1 parameter which is dilation factor. 
%                       
%                       options.KernelPars: a row vector. Minimum size of it is 0 (when no parameters is needed for kernel function)
%                                           and maximum size of it is 2 (when kernel function has 2 parameters)
%
%
% Output:
%           k:      kernel function value (a real scalar)
% 
% 
% Sample use:    
%       k = kernel(u,v,options);     % computes dot product in feature space
% 
% 
% 
% Bugs: (1) Before using this function, you should determine suitable values for options.KernelType and options.KernelPars 
%           variables otherwise the default values will be used by function.
%       (2) If you use spherically normalized kernels, you'd better already normalize the training data to zero mean and unit variance then set the spherical
%           normalization parameter to 1.
%
% 
% 
%   More details can be found in:
%   
%   S. Meshgini, A. Aghagolzadeh, H. Seyedarabi, "Face recognition using 
%   Gabor-based direct linear discriminant analysis and support vector machine," 
%   Computers & Electrical Engineering, vol. 39, no. 3, pp. 727-745, 2013.
% 
% 
% 
%   (C)	Saeed Meshgini, Ph.D.
%       University of Tabriz



% checking the correct use of input arguments:

if (~exist('options','var'))
   options.KernelType = 'linear';
end

% checking the same dimensionality of input vectors:

p = length(u);
q = length(v);
if p ~= q
    error('dimension of two vectors must be the same.')
end

% kernels:

if ~isfield(options,'KernelType')
    options.KernelType = 'linear';          % default kernel function is linear
end

switch lower(options.KernelType)
    case {'linear'}                 
%         k = u.'*v;                          % u'*v
        k = (u')*v;
    case {'poly'}            
        if ~isfield(options,'KernelPars')
            options.KernelPars = 2;         % default value for degree of polynomial is 2
        end
        k = ((u')*v).^options.KernelPars;
%         k = (u.'*v)^options.KernelPars;     % (u'*v)^n
    case {'polyplus'}                
        if ~isfield(options,'KernelPars')
            options.KernelPars = 2;         % default value for degree of polynomial is 2
        end
        k = ((u')*v+1).^options.KernelPars;
%         k = (u.'*v+1)^options.KernelPars;   % (u'*v+1)^n
    case {'sphnorm_poly'}      
        if ~isfield(options,'KernelPars')
            options.KernelPars = [2 1];     % default value for degree of polynomial is 2 and default parameter for spherical normalization is 1
        end
        
        k = (1/2^options.KernelPars(1))*...
             ((           ((u')*v + options.KernelPars(2)^2)./...
                 sqrt( (sum(u.^2,1)' + options.KernelPars(2)^2) *...
                       (sum(v.^2,1) + options.KernelPars(2)^2)   ) )+1).^options.KernelPars(1);
       
%         k = (1/2^options.KernelPars(1))*...
%              ((        (u.'*v+options.KernelPars(2)^2)/...
%                  sqrt( (u.'*u+options.KernelPars(2)^2)*...
%                        (v.'*v+options.KernelPars(2)^2) )   )+1)^options.KernelPars(1);
                                            % ((u'*v+d^2)/sqrt((u'*u+d^2)(v'*v+d^2))+1)^n/2^n
    case {'rbf'}        
        if ~isfield(options,'KernelPars')   % default value for sigma is 1
            options.KernelPars = 1; 
        end
        k = zeros(size(u,2),size(u,2));
        for i=1:size(u,2)
            for j=1:size(v,2)
                dTmp = u(:,i)-v(:,j);
                k(i,j) = exp((-dTmp'*dTmp)/(2*options.KernelPars^2));
            end
        end
%         k = exp((-(u-v).'*(u-v))/(2*options.KernelPars^2));
                                            % e^{-(|u-v|^2)/2(sigma)^2}
    case {'wave'}        
        if ~isfield(options,'KernelPars')   % default value for sigma is 1
            options.KernelPars = 2 ;        % default value for dilation factor is 2 and default mother wavelet is morlet function
        end 
%         if strcmp(options.KernelPars(2),'morl')
            k = zeros(size(u,2),size(u,2));
            for i=1:size(u,2)
                for j=1:size(v,2)
                    dTmp = u(:,i)-v(:,j);
                    k(i,j) = prod(cos(1.75*dTmp/options.KernelPars).*exp(-(dTmp.^2)/(2*options.KernelPars^2)));
                end
            end
%             pro = 1;
%             for i = 1:p
%                 pro = pro*(cos(1.75*((u(i)-v(i))/options.KernelPars))*exp(-((u(i)-v(i))^2)/(2*options.KernelPars^2)));  
%             end
%             k = pro;                        % Pro(psi(ui-vi)/a), psi(x) = cos(1.75x)exp(-x^2/2)
%         elseif strcmp(options.KernelPars(2),'mexh')
%             pro = 1;
%             for i = 1:p
%                 pro = pro*(((2/sqrt(3))*pi^(-0.25))*(1-((u(i)-v(i))/options.KernelPars(1))^2)*exp((-((u(i)-v(i))/options.KernelPars(1))^2)/2));
%             end
%             k = pro;
%         elseif strcmp(options.KernelPars(2),'haar')
%             pro = 1;
%             for i = 1:p
%                 if (((u(i)-v(i))/options.KernelPars(1))> = 0)&&(((u(i)-v(i))/options.KernelPars(1))<0.5)
%                     pro = pro*1;
%                 elseif (((u(i)-v(i))/options.KernelPars(1))> = 0.5)&&(((u(i)-v(i))/options.KernelPars(1))<1)
%                     pro = pro*(-1);
%                 else
%                     pro = pro*0;
%                 end
%             end
%             k = pro;
%         else
%             error('use one of these wavelets ("morl" or "mexh" or "haar") as valid wavelet types')
%         end
    otherwise
        error('KernelType does not exist!');
end
end
