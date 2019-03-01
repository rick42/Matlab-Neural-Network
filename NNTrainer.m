classdef NNTrainer < handle
    properties
        N;
        J;
        X;
        Y;
        learnRate = 25;
        iterations = 25000000000;
        bestJ;
        bestParam;
        result;
    end
    methods
        % CONSTRUCTOR
        function M = NNTrainer(N)
            if nargin == 1
                M.N = N;
            else
                error('Invalid number of inputs');
            end
        end
        
        function train(self,X,Y)
            % Make an internal variable for the callback function:
            self.X = X;
            self.Y = Y;
            
            % Make empty list to store costs:
            self.J = [];
            
            self.bestParam = self.N.getParams();
            self.bestJ = self.N.costFunction(X,Y);
            self.J(1) = self.bestJ;
            
            fprintf('Iteration: 0        Cost: %f\n',self.bestJ);
            
            
            
            for i = 1:self.iterations
                self.N.batchGradientDescent(X,Y,self.learnRate);
                
                if i == 1
                    gradBefore = self.N.getGradient(X,Y);
                end
                
                self.J(i+1) = self.N.costFunction(X,Y);
                if self.J(i+1) < self.bestJ
                    self.bestJ = self.J(i+1);
                    self.bestParam = self.N.getParams();
                else
                    self.J(i+1) = self.bestJ;
                end
                %if(mod(i,50) == 0)
                if(mod(i,1) == 0)
                    fprintf('Iteration: %d        Cost: %f\n',i,self.bestJ);
                end
                %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                if self.bestJ < .015
                    save i;
                    break
                end
                %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            end
            
            gradAfter = self.N.getGradient(X,Y);
            
            gradDif = gradAfter - gradBefore;
            self.result = self.bestParam;
          
%             self.result = fminunc(costFuncHandle,params0,options);
          
        end
    end
end