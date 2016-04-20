classdef PERCEPTRON < handle
    %PERCEPTRON Implements a multi-layer perceptron with sigmoid tfn
    % Copyright Anton Tkachev 2015
    % Modified by Brian Bleakley 2016
    %% Properties of the perceptron
    properties
        layer;    % array that defines the number of neurons in each layer
        nLayers;  % total number of layers  
        nTrans;   % total number of transitions between the layers
        weight;   % cell of neurons' weights matrices
        alpha;    % sigmoid function coefficient
        divergence; % boolean value; whether the network has diverged
    end
    
    %% Methods of the perceptron
    methods
        %% Constructor
        function obj = PERCEPTRON(layers_vector)
            obj.alpha = 2.5;
            obj.layer = layers_vector;
            obj.nLayers = length(layers_vector);
            obj.nTrans = length(layers_vector) - 1;
            obj.weight = cell(obj.nTrans,1);
            obj.divergence = 0;
            
            a = 0.5;    % bounds for weights random initialization
            for i = 1 : obj.nTrans
                obj.weight{i} = 2*a*rand(obj.layer(i+1)+1,obj.layer(i)+1) - a;
            end
        end
        
        %% Forward neural network calculation
        function out = forward(obj,input_col_vector)
            n = obj.nTrans;%nLayers - 1
            A = cell(obj.nLayers,1);
            
            A{1} = [input_col_vector;1];%bias
            for i = 1 : n - 1
                A{i+1} = PERCEPTRON.tfn(obj.weight{i}*A{i},obj.alpha);
                A{i+1}(end) = 1;%bias
            end
            A{n+1} = obj.weight{n}*A{n};
            
            A{n+1} = A{n+1}(1:(end-1));%remove the last element (would be bias node)
            
            out = PERCEPTRON.tfn(A{obj.nLayers},obj.alpha);%same as n+1
        end
        
        %% Error back propagation. Single sample
        function err = backprop(obj,input,desired_output,eta)
            n = obj.nTrans;
            O = cell(obj.nLayers,1);
            
            O{1} = [input;1];%bias
            for i = 1 : n - 1
                O{i+1} = PERCEPTRON.tfn(obj.weight{i}*O{i},obj.alpha);
                O{i+1}(end) = 1;%bias
            end
            O{n+1} = obj.weight{n}*O{n};
            %O{n+1} = O{1:(end-1)};%remove the last element (would be bias node)
            
            O = flip(O);
            T = [desired_output;1];%bias
            W = flip(obj.weight);
            delta = cell(obj.nTrans,1);
            
            err = (T - O{1});
            delta{1} = -2*obj.alpha*O{1}.*(1 - O{1}).*err;
            
            for i = 2 : obj.nTrans
                delta{i} = 2*obj.alpha*O{i}.*(1 - O{i}).*(W{i-1}.'*delta{i-1});
            end
            
            for i = 1 : obj.nTrans
                W{i} = W{i} - eta*delta{i}*O{i+1}.';
            end
            obj.divergence = max(isnan(W{1}));
            obj.weight = flip(W);
        end
    end
    
    %% Transfer function methods
    methods(Static, Access = private)
        %% Exponential sigmoid transfer function
        function out = tfn(input_vector,input_alpha)
            n = length(input_vector);
            out = zeros(n,1);
            for i = 1 : n
                out(i) = 1/(1 + exp(-2*input_alpha*input_vector(i)));
            end
        end
    end
    
end
