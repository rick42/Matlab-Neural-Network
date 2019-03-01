classdef myMultilayerNN < handle
    properties
        inputLayerSize = 3;
        hiddenLayerSize = [4 2];
        outputLayerSize = 1;
        allLayerSize;
        b;
        W;
        a;
        z;
        yHat;
        delta;
        dJdW;
        dJdb;
        lamda = 0;
        alpha;
    end
    methods
        function M = myMultilayerNN(inLayerSize, hidLayerSize, outLayerSize)
            if nargin == 3
                M.inputLayerSize = inLayerSize;
                M.hiddenLayerSize = hidLayerSize;
                M.outputLayerSize = outLayerSize;
                M.setupAllLayerSize;
                % Create a weight matrix for each hidden layer and the
                % output layer
                M.setupWeights;
                M.setupBias;
                M.z = cell(1,length(M.allLayerSize));
                M.a = cell(1,length(M.allLayerSize));
                M.delta = cell(1,length(M.allLayerSize));
                M.dJdW = cell(1,length(M.W));
                M.dJdb = cell(1,length(M.b));
                 
            elseif nargin == 0
            else
                error('Invalid number of inputs');
            end
        end
        
        function setupAllLayerSize(self)
            temp = zeros(1,length(self.hiddenLayerSize)+2);
            temp(1) = self.inputLayerSize;
            temp(2:end-1) = self.hiddenLayerSize;
            temp(end) = self.outputLayerSize;
            self.allLayerSize = temp;
        end
        
        function setupWeights(self)
            self.W = cell(1,length(self.allLayerSize)-1);
            for i = 1:length(self.allLayerSize)-1
                self.W{i} = normrnd(0,0.20,self.allLayerSize(i+1),self.allLayerSize(i));
            end
        end
        
        function setupBias(self)
            self.b = cell(1,length(self.allLayerSize)-1);
            for i = 1:length(self.allLayerSize)-1
                self.b{i} = normrnd(0,0.20,self.allLayerSize(i+1),1);
            end
        end
        
        function sig = sigmoidFunc(self,z)
            expTemp = exp(-z);
            sig = 1./(1+expTemp);
        end
        
        function sigPrime = sigmoidPrime(self,z)
            sigPrime = self.sigmoidFunc(z).*(1-self.sigmoidFunc(z));
        end
        
        function res = rectiLinearFunc(self,z)
            res = z;
            res(res < 0) = 0;
        end

        function res = rectiLinearPrime(self,z)
            res = z;
            res(res <= 0) = 0;
            res(res > 0) = 1;
        end
        
        function yHat = forward(self,X)
            self.a{1} = X;
            for i = 2:length(self.allLayerSize)
                self.z{i} = self.calcZ(i-1);
                self.a{i} = self.sigmoidFunc(self.z{i});
            end
            yHat = self.a{end};
        end
        
        function z = calcZ(self,layer)
            dotProd = self.W{layer}*self.a{layer};
            z = dotProd + self.b{layer};
        end
        
        function params = getParams(self)
            params = [];
            for i = 1:length(self.W)
                tempMatrix = self.W{i};
                flatMatrix = self.flattenMat(tempMatrix);
                params = cat(2,params,flatMatrix);
            end
            for i = 1:length(self.b)
                params = cat(2,params,self.b{i}');
            end
        end
        
        function flatMat = flattenMat(self,M)
            flatMat = zeros(1,numel(M));
            count = 0;
            for i = 1:size(M,1)
                for j = 1:size(M,2)
                    count = count + 1;
                    flatMat(count) = M(i,j);
                end
            end
        end
        
        function setParams(self,params)
            if length(params) ~= length(self.getParams)
                error('Input for method setParams has incorrect length');
            end
            
            currentMatStart = 0;
            currentMatEnd = 0;
            
            for i = 1:length(self.W)
                currentMatStart = currentMatEnd + 1;
                currentMatEnd = currentMatEnd + numel(self.W{i});
                self.W{i} = reshape(params(currentMatStart:currentMatEnd),...
                                        size(self.W{i},2),size(self.W{i},1))';
            end
            
            for i = 1:length(self.b)
                currentMatStart = currentMatEnd + 1;
                currentMatEnd = currentMatEnd + numel(self.b{i});
                self.b{i} = params(currentMatStart:currentMatEnd)';
            end
        end
        
        function J = costFunction(self,X,Y)
            self.yHat = self.forward(X);
            J = self.getSqOfErrorsTerm(Y) + self.getWeightDecayTerm;
        end
        
        function term = getSqOfErrorsTerm(self,Y)
            sum = 0;
            for i = 1:size(Y,2)
                temp = norm(self.yHat(:,i) - Y(:,i));
                temp = temp^2 * .5;
                sum = sum + temp;
            end
            term = sum/size(Y,2);
        end
        
        function term = getWeightDecayTerm(self)
            sum = 0;
            for l = 1:length(self.allLayerSize)-1
                for i = 1:self.allLayerSize(l)
                    for j = 1:self.allLayerSize(l+1)
                        sum = self.W{l}(j,i)^2 + sum;                        
                    end
                end
            end
            term = self.lamda * .5 * sum;
        end
        
        function backPropagation(self,X,Y)
            self.yHat = self.forward(X);
            
            self.getDeltas(Y);
            self.getPartialDerivative;
%             for i = 1:length(self.delta)
%                 for j = 1:numel(self.delta{i})
%                     if ~isfinite(self.delta{i}(j))
%                         disp('Not finite result');
%                     end
%                 end
%             end
%             
%             self.dJdW{1} = X' * self.delta{1};
%             for i = 2:length(self.dJdW)
%                 self.dJdW{i} = self.A{i-1}' * self.delta{i};
%             end
            
%             for i = 1:length(self.dJdW)
%                 for j = 1:numel(self.dJdW{i})
%                     if ~isfinite(self.dJdW{i}(j))
%                         error('Not finite result');
%                     end
%                 end
%             end
        end
        
        function gradients = getGradient(self,X,Y)
            %self.backPropagation(X,Y);
            
            gradients = [];
            for i = 1:length(self.dJdW)
                tempMatrix = self.dJdW{i};
                flatMatrix = self.flattenMat(tempMatrix);
                gradients = cat(2,gradients,flatMatrix);
            end
            for i = 1:length(self.dJdb)
                gradients = cat(2,gradients,self.dJdb{i}');
            end
        end
        
        function getDeltas(self,Y)
            self.delta{end} = -(Y-self.a{end}).*self.sigmoidPrime(self.z{end});
            for i = length(self.allLayerSize)-1:-1:2
                self.delta{i} = ((self.W{i})'*self.delta{i+1}).*self.sigmoidPrime(self.z{i});
            end
        end
        
        function getPartialDerivative(self)
            for i = 1:length(self.W)
                self.dJdW{i} = self.delta{i+1}*self.a{i}';
                self.dJdb{i} = self.delta{i+1};
            end
        end
        
        function batchGradientDescent(self,X,Y,learnRate)
            numSamples = size(Y,2);
            stepW = cell(1,length(self.W));
            stepb = cell(1,length(self.b));
                       
            for i = 1:length(stepW)
                stepW{i} = zeros(size(self.W{i}));
            end
            for i = 1:length(stepb)
                stepb{i} = zeros(size(self.b{i}));
            end
            
            for i = 1:numSamples
                singleX = X(:,i);
                singleY = Y(:,i);
                self.backPropagation(singleX,singleY);
                for j = 1:length(stepW)
                    stepW{j} = stepW{j} + self.dJdW{j};
                    stepb{j} = stepb{j} + self.dJdb{j};
                end
            end
            
            for i = 1:length(stepW)
                learnTermW = ((1/numSamples) .* stepW{i}) + (self.lamda .* self.W{i});
                learnTermb = (1/numSamples) .* stepb{i};
                self.W{i} = self.W{i} - learnRate .* learnTermW;
                self.b{i} = self.b{i} - learnRate .* learnTermb;
            end
        end

    end
end