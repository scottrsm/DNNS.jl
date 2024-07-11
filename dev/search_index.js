var documenterSearchIndex = {"docs":
[{"location":"#DNNS.jl-Documentation","page":"DNNS.jl Documentation","title":"DNNS.jl Documentation","text":"","category":"section"},{"location":"","page":"DNNS.jl Documentation","title":"DNNS.jl Documentation","text":"CurrentModule = DNNS","category":"page"},{"location":"#Overview","page":"DNNS.jl Documentation","title":"Overview","text":"","category":"section"},{"location":"","page":"DNNS.jl Documentation","title":"DNNS.jl Documentation","text":"Below is a collection of structures and functions which are useful to create (D)eep (N)eural (N)etworks. One of the features of this package is the ability to do Automatic Differentiation. This is accomplished via the structure, AD. The basic arithmatic functions as well as the power, abs, log, exponential, and all of the trigonometric functions from the base  package are overloaded by this package to work with AD \"numbers\".","category":"page"},{"location":"","page":"DNNS.jl Documentation","title":"DNNS.jl Documentation","text":"Additionally, this package adds a notion of a piece-wise linear function via the struct, PWL. This structure, as well as are the structures DLayer and DNN, has been instrumented so  that it can also serve as a function.","category":"page"},{"location":"","page":"DNNS.jl Documentation","title":"DNNS.jl Documentation","text":"Structures:\nAD: A structure used to implement Automatic Differentiation.\nPWL: A structure representing a piece-wise linear function.\nDLayer: A structure representing a single layer of a DNN.\nDNN: A structure representing a Deep Neural Net which consists of        Some number of DLayers.\nFunctions\nL1\nNon-linear Activation Functions\nsigmoid1\nsigmoid2\nsigmoid3\nsoftmax\nrelu\nrelur\nAssociated Learning Functions\nloss\nfit","category":"page"},{"location":"#Data-Structures","page":"DNNS.jl Documentation","title":"Data Structures","text":"","category":"section"},{"location":"","page":"DNNS.jl Documentation","title":"DNNS.jl Documentation","text":"AD","category":"page"},{"location":"#DNNS.AutoDiff.AD","page":"DNNS.jl Documentation","title":"DNNS.AutoDiff.AD","text":"AD{T}\n\nAutomatic differentiation structure.\n\nFields\n\nv :: T – The value of this structure.\nd :: T – The derivative at this value.\n\n\n\n\n\n","category":"type"},{"location":"","page":"DNNS.jl Documentation","title":"DNNS.jl Documentation","text":"PWL","category":"page"},{"location":"#DNNS.PWLF.PWL","page":"DNNS.jl Documentation","title":"DNNS.PWLF.PWL","text":"PWL{T}\n\nA structure representing a piece-wise linear function on the Real line.\n\nIn practice, one uses one of two outer constructors to create a PWL struct.\n\nType Constraints\n\nT <: Number\nThe type T must have a total ordering.\n\nFields\n\nxs :: Vector{T}  – The \"x\" values.\nys :: Vector{T}  – The \"y\" values.\nds :: Vector{T}  – The \"slopes\" of the segments. Including the left most and right most slopes.\nn  :: Int        – The number of \"x,y\" values.\n\nPublic Constructors\n\nPWL(xs::Vector{T}, y::T, ds::Vector{T}) \n\nxs – The x coordinates in ascending order – no duplicates.\ny  – The value of y corresponding to the first entry in xs.\nds – The slopes of all \"x\" intervals as well as the \"left\" slope of the first         point and the \"right\" slope of the last point.\n\nPWL(xs::Vector{T}, ys::Vector{T}, ds::Vector{T})\n\nxs – The x coordinates in ascending order – no duplicates.\nys – The y coordinates corresponding to each x value.\nds – A 2-Vector consisting of the \"left\" slope of the first point and the \"right\"         slope of the last point.\n\nExamples\n\njulia> # Create the same (in behavior) Piecewise linear functions in two ways:\njulia> pw1 = PWL([1.0, 2.0, 3.0], [2.0, 3.0, 4.0], [0.0, 5.0])\n\njulia> pw2 = PWL([1.0, 2.0, 3.0], 2.0, [0.0, 1.0, 1.0, 5.0])\n\njulia> pw1(2.5)\n3.5\n\njulia> pw2(2.5)\n3.5\n\n\n\n\n\n","category":"type"},{"location":"","page":"DNNS.jl Documentation","title":"DNNS.jl Documentation","text":"DLayer","category":"page"},{"location":"#DNNS.DLayer","page":"DNNS.jl Documentation","title":"DNNS.DLayer","text":"DLayer{T<:Number}\n\nA structure representing one layer of a neural net. \n\nType Constraints\n\nT <: Number\nThe type T must have a total ordering.\n\nFields\n\nM    :: Matrix{AD{T}}    – The \"x\" values.\nb    :: Vector{AD{T}}    – The \"y\" values.\nop   :: Function         – The \"slopes\" of the segments.\ndims :: Tuple{Int, Int}  – The number of \"x,y\" values.\n\nPublic Constructors\n\nDLayer(Mn::Matrix{T}, bn::Vector{T}, opn::Function)\n\nMn – A MxNweight matrix.\nbn – A Ndimensional bias vector.\nop – The non-linear threshold function.\n\n\n\n\n\n","category":"type"},{"location":"","page":"DNNS.jl Documentation","title":"DNNS.jl Documentation","text":"DNN","category":"page"},{"location":"#DNNS.DNN","page":"DNNS.jl Documentation","title":"DNNS.DNN","text":"DNN{T<:Number}\n\nA structure representing one layer of a neural net. \n\nType Constraints\n\nT <: Number\nThe type T must have a total ordering.\n\nFields\n\nlayers :: Vector{DLayer{T} – The neural net layers.\n\nPublic Constructors\n\nDNN(ls::Vector{DLayer{T}}\n\nls – A vector of DLayer.\n\n\n\n\n\n","category":"type"},{"location":"","page":"DNNS.jl Documentation","title":"DNNS.jl Documentation","text":"(PWL)(::T) where {T<:Number}","category":"page"},{"location":"#DNNS.PWLF.PWL-Tuple{T} where T<:Number","page":"DNNS.jl Documentation","title":"DNNS.PWLF.PWL","text":"(PWL{T})(x::T) where {T<:Number}\n\nUses the structure PWL as a piece-wise linear function. \n\nType Constraints\n\nT <: Number\n\nArguments\n\nx :: T  – An input value.\n\nReturn\n\n:: T\n\n\n\n\n\n(PWL{T})(x::AD{T}) where {T<:Number}\n\nUses the structure PWL as a piece-wise linear function. \n\nType Constraints\n\nT <: Number\n\nArguments\n\nx :: AD{T}  – An AutoDiff value.\n\nReturn\n\n:: AD{T}\n\n\n\n\n\n","category":"method"},{"location":"","page":"DNNS.jl Documentation","title":"DNNS.jl Documentation","text":"(DLayer)(::AD{T}) where {T <: Number}","category":"page"},{"location":"#DNNS.DLayer-Union{Tuple{DNNS.AutoDiff.AD{T}}, Tuple{T}} where T<:Number","page":"DNNS.jl Documentation","title":"DNNS.DLayer","text":"(L::DLayer{T})(x::AD{T}) where {T <: Number}\n\nIf (M,N) = L.dims, then we may treat the structure, DLayer, as a function: cal R^m mapsto cal R^n .\n\nTakes input x and passes it through the layer.\n\nType Constraints\n\nT <: Number\n\nArguments\n\nx :: AD{T}  – An input value of dimension N.\n\nReturn\n\n::Vector{AD{T}} of dimension M.\n\n\n\n\n\n","category":"method"},{"location":"","page":"DNNS.jl Documentation","title":"DNNS.jl Documentation","text":"(DNN)(::AbstractVector{T}) where {T <: Number}","category":"page"},{"location":"#DNNS.DNN-Union{Tuple{AbstractVector{T}}, Tuple{T}} where T<:Number","page":"DNNS.jl Documentation","title":"DNNS.DNN","text":"(dnn::DNN{T})(x::AbstractVector{T}) where {T <: Number}\n\nLet N = DNN.ls[1].dims[2] and M = DNN.ls[end].dims[1], then here we treat the structure DNN as a function: cal R^N mapsto cal R^M Takes input x and passes it through each of the layers of DNN.\n\nType Constraints\n\nT <: Number\n\nArguments\n\nx :: T  – An input value of dimension N.\n\nReturn\n\n::Vector{AD{T}} of dimension M.\n\n\n\n\n\n","category":"method"},{"location":"#Non-linear-Activation-Functions","page":"DNNS.jl Documentation","title":"Non-linear Activation Functions","text":"","category":"section"},{"location":"","page":"DNNS.jl Documentation","title":"DNNS.jl Documentation","text":"sigmoid1","category":"page"},{"location":"#DNNS.UtilFunc.sigmoid1","page":"DNNS.jl Documentation","title":"DNNS.UtilFunc.sigmoid1","text":"sigmoid1(x::AD{T})\n\nImplements an AD version of the standard \"exponential\" sigmoid function.\n\nType Constraints\n\nT <: Number\n\nArguments\n\nx   :: AD{T}  – The AD input value.\n\nReturn\n\n::AD{T} – The output AD value/derivative.\n\n\n\n\n\n","category":"function"},{"location":"","page":"DNNS.jl Documentation","title":"DNNS.jl Documentation","text":"sigmoid2","category":"page"},{"location":"#DNNS.UtilFunc.sigmoid2","page":"DNNS.jl Documentation","title":"DNNS.UtilFunc.sigmoid2","text":"sigmoid2(x::AD{T})\n\nImplements an AD version of the standard \"tanh\" sigmoid function.\n\nType Constraints\n\nT <: Number\n\nArguments\n\nx   :: AD{T}  – The AD input value.\n\nReturn\n\n::AD{T} – The output AD value/derivative.\n\n\n\n\n\n","category":"function"},{"location":"","page":"DNNS.jl Documentation","title":"DNNS.jl Documentation","text":"sigmoid3","category":"page"},{"location":"#DNNS.UtilFunc.sigmoid3","page":"DNNS.jl Documentation","title":"DNNS.UtilFunc.sigmoid3","text":"sigmoid3(x::AD{T})\n\nImplements an AD version of the standard \"arctan\" sigmoid function.\n\nType Constraints\n\nT <: Number\n\nArguments\n\nx   :: AD{T}  – The AD input value.\n\nReturn\n\n::AD{T} – The output AD value/derivative.\n\n\n\n\n\n","category":"function"},{"location":"","page":"DNNS.jl Documentation","title":"DNNS.jl Documentation","text":"relu","category":"page"},{"location":"#DNNS.UtilFunc.relu","page":"DNNS.jl Documentation","title":"DNNS.UtilFunc.relu","text":"relu(x::AD{T})\n\nImplements an AD version of the standard relu function.\n\nType Constraints\n\nT <: Number\n\nArguments\n\nx   :: AD{T}  – The AD input value.\n\nReturn\n\n::AD{T} – The output AD value/derivative.\n\n\n\n\n\n","category":"function"},{"location":"","page":"DNNS.jl Documentation","title":"DNNS.jl Documentation","text":"relur","category":"page"},{"location":"#DNNS.UtilFunc.relur","page":"DNNS.jl Documentation","title":"DNNS.UtilFunc.relur","text":"relur(x::AD{T})\n\nImplements an AD version of a modified version of the relu function. The modification is that while the value of the relur is the same as relu, its derivative is not. The value of the derivative is 0 or 1, however the boundary moves randomly around the natural input boundary of 0,\n\nT <: Number\n\nArguments\n\nx   :: AD{T}  – The AD input value.\n\nReturn\n\n::AD{T} – The output AD value/derivative.\n\n\n\n\n\n","category":"function"},{"location":"","page":"DNNS.jl Documentation","title":"DNNS.jl Documentation","text":"softmax","category":"page"},{"location":"#DNNS.UtilFunc.softmax","page":"DNNS.jl Documentation","title":"DNNS.UtilFunc.softmax","text":"softmax(x::Vector{T} [, τ=one(T)])\n\nImplements the softmax function.\n\nType Constraints\n\nT <: Number\n\nArguments\n\nx :: Vector{T}  – The AD input vector.\nτ :: T          – The \"temperature\" parameter. \n\nReturn\n\n::Vector{T} – The output AD vector.\n\n\n\n\n\nsoftmax(x::Vector{AD{T}} [, τ=one(T)])\n\nImplements an AD version of the softmax function.\n\nType Constraints\n\nT <: Number\n\nArguments\n\nx :: Vector{AD{T}}  – The AD input vector.\nτ :: T              – The \"temperature\" parameter. \n\nReturn\n\n::Vector{AD{T}} – The output AD vector.\n\n\n\n\n\n","category":"function"},{"location":"#Functions","page":"DNNS.jl Documentation","title":"Functions","text":"","category":"section"},{"location":"","page":"DNNS.jl Documentation","title":"DNNS.jl Documentation","text":"loss","category":"page"},{"location":"#DNNS.loss","page":"DNNS.jl Documentation","title":"DNNS.loss","text":"loss(dnn, X, Y)\n\nComputes the loss of the neural network given inputs, X, and outputs Y.\n\nType Constraints\n\nT <: Number\n\nArguments\n\ndnn :: DNN{T}     – A DNN layer.\nX   :: Matrix{T}  – The matrix of input values.\nY   :: Matrix{T}  – The matrix of output values.\n\nReturn\n\n::AD{T} – The loss of the network\n\n\n\n\n\n","category":"function"},{"location":"","page":"DNNS.jl Documentation","title":"DNNS.jl Documentation","text":"fit","category":"page"},{"location":"#DNNS.fit","page":"DNNS.jl Documentation","title":"DNNS.fit","text":"fit(dnn, X, Y)\n\nAdjusts the parameters of neural network, dnn, to get the best fit of  the data: X, Y. The parameters of the network are all paris of  matrices and biases for each layer in the network.\n\nType Constraints\n\nT <: Number\n\nArguments\n\ndnn :: DNN{T}     – A DNN layer.\nX   :: Matrix{T}  – The matrix of input values.\nY   :: Matrix{T}  – The matrix of output values.\n\nReturn\n\n::nothing\n\n\n\n\n\n","category":"function"},{"location":"#Index","page":"DNNS.jl Documentation","title":"Index","text":"","category":"section"},{"location":"","page":"DNNS.jl Documentation","title":"DNNS.jl Documentation","text":"","category":"page"}]
}
