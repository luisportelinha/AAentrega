using DelimitedFiles, Statistics, Random, Plots, Flux, Flux.Losses
using Random: seed!
using Flux: params
using PrettyTables
using ScikitLearn

@sk_import svm: SVC 
@sk_import tree: DecisionTreeClassifier 
@sk_import neighbors: KNeighborsClassifier

function oneHotEncoding(feature::AbstractArray{<:Any,1}, classes::AbstractArray{<:Any,1})
    encoded = falses(length(feature), length(classes))
    for (index, class) in enumerate(classes)
        encoded[:, index] .= (feature .== class)
    end
    encoded
end

oneHotEncoding(feature::AbstractArray{<:Any,1}) = oneHotEncoding(feature, unique(feature))
oneHotEncoding(feature::AbstractArray{Bool,1}) = reshape(feature, :, 1)

function calculateMinMaxNormalizationParameters(dataset::AbstractArray{<:Real,2})
    (minimum(dataset, dims=1), maximum(dataset, dims=1))
end

function calculateZeroMeanNormalizationParameters(dataset::AbstractArray{<:Real,2})
    (mean(dataset, dims=1), std(dataset, dims=1))
end

function normalizeMinMax!(dataset::AbstractArray{<:Real,2}, normParams::NTuple{2, AbstractArray{<:Real,2}})
    for col in 1:size(dataset, 2)
        delta = normParams[2][col] - normParams[1][col]
        if delta != 0
            dataset[:, col] .= (dataset[:, col] .- normParams[1][col]) ./ delta
        else
            dataset[:, col] .= 0.0
        end
    end
end

normalizeMinMax!(dataset::AbstractArray{<:Real,2}) = normalizeMinMax!(dataset, calculateMinMaxNormalizationParameters(dataset))

function normalizeMinMax(dataset::AbstractArray{<:Real,2})
    copy_dataset = copy(dataset)
    normalizeMinMax!(copy_dataset, calculateMinMaxNormalizationParameters(dataset))
    copy_dataset
end

normalizeMinMax(dataset::AbstractArray{<:Real,2}) = normalizeMinMax(dataset, calculateMinMaxNormalizationParameters(dataset));


function normalizeZeroMean!(dataset::AbstractArray{<:Real,2}, normParams::NTuple{2, AbstractArray{<:Real,2}})
    for col in 1:size(dataset, 2)
        if normParams[2][col] != 0
            dataset[:, col] .= (dataset[:, col] .- normParams[1][col]) ./ normParams[2][col]
        else
            dataset[:, col] .= 0.0
        end
    end
end

normalizeZeroMean!(dataset::AbstractArray{<:Real,2}) = normalizeZeroMean!(dataset, calculateZeroMeanNormalizationParameters(dataset))

function normalizeZeroMean(dataset::AbstractArray{<:Real,2})
    copy_dataset = copy(dataset)
    normalizeZeroMean!(copy_dataset, calculateZeroMeanNormalizationParameters(dataset))
    copy_dataset
end

function classifyOutputs(outputs::AbstractArray{<:Real,2}; threshold::Real=0.5)
    if size(outputs, 2) == 1
        return outputs .>= threshold
    else
        max_indices = argmax(outputs, dims=2)
        classified = falses(size(outputs))
        for i in 1:length(max_indices)
            classified[i, max_indices[i]] = true
        end
        classified
    end
end

accuracy(outputs::AbstractArray{Bool,1}, targets::AbstractArray{Bool,1}) = mean(outputs.==targets);

function accuracy(targets::AbstractArray{Bool,2}, outputs::AbstractArray{Bool,2})
    @assert size(targets, 2) == size(outputs, 2) "The number of columns in targets and outputs must be the same."

    # Handle single-column case directly
    if size(targets, 2) == 1
        return mean(targets[:] .== outputs[:])
    end

    # Calculate accuracy for multi-class outputs
    correctClassifications = all(targets .== outputs, dims=2)
    return mean(correctClassifications)
end

function accuracy(outputs::AbstractArray{<:Real,1}, targets::AbstractArray{Bool,1}; threshold::Real=0.5)
    classified = outputs .>= threshold
    accuracy(classified, targets)
end

function accuracy(outputs::AbstractArray{<:Real,2}, targets::AbstractArray{Bool,2}; threshold::Real=0.5)
    classified = classifyOutputs(outputs, threshold=threshold)
    accuracy(classified, targets)
end

function buildClassANN(numInputs::Int, topology::AbstractArray{<:Int,1}, numOutputs::Int;
    transferFunctions::AbstractArray{<:Function,1}=fill(σ, length(topology))) 

    ann = Chain();
    numInputsLayer = numInputs;
    iteration = 1;
    for numOutputsLayer = topology
        ann = Chain(ann..., Dense(numInputsLayer, numOutputsLayer,  transferFunctions[iteration]));
        numInputsLayer = numOutputsLayer;
        iteration += 1;
    end
    if (numOutputs == 1)
        ann = Chain(ann..., Dense(numInputsLayer, 1, σ));
    else
        ann = Chain(ann..., Dense(numInputsLayer, numOutputs, identity));
        ann = Chain(ann..., softmax);
    end
    return ann;
end

function holdOut(N::Int, p::Real)
    @assert 0 <= p <= 1
    idx = randperm(N)
    split = round(Int, N * p)
    (idx[1:N-split], idx[N-split+1:end])
end

function holdOut(N::Int, pVal::Real, pTest::Real)
    @assert pVal + pTest <= 1
    train_val_idx, test_idx = holdOut(N, pTest)
    train_idx, val_idx = holdOut(length(train_val_idx), pVal / (1 - pTest))
    (train_val_idx[train_idx], train_val_idx[val_idx], test_idx)
end

function trainClassANN(topology::AbstractArray{<:Int,1},
                       trainingDataset::Tuple{AbstractArray{<:Real,2}, AbstractArray{Bool,2}};
                       validationDataset::Tuple{AbstractArray{<:Real,2}, AbstractArray{Bool,2}} = 
                       (zeros(0, 0), falses(0, 0)),
                       testDataset::Tuple{AbstractArray{<:Real,2}, AbstractArray{Bool,2}} = 
                       (zeros(0, 0), falses(0, 0)),
                       transferFunctions::AbstractArray{<:Function,1} = fill(σ, length(topology)),
                       maxEpochs::Int = 1000, minLoss::Real = 0.0, learningRate::Real = 0.01,
                       maxEpochsVal::Int = 20, showText::Bool = false)
    
    # Create the ANN using a different setup
    layers = [Dense(size(trainingDataset[1], 2), topology[1], transferFunctions[1])]
    append!(layers, [Dense(topology[i], topology[i + 1], transferFunctions[i + 1]) for i in 1:length(topology) - 1])
    append!(layers, [Dense(topology[end], size(trainingDataset[2], 2), identity)])
    ann = Chain(layers...)
    
    # Define loss function based on the output layer
    lossFunction = size(trainingDataset[2], 2) == 1 ? Losses.binarycrossentropy : Losses.crossentropy
    
    # Training loop with early stopping and validation loss tracking
    bestModel, bestLoss = deepcopy(ann), Inf
    for epoch in 1:maxEpochs
        Flux.train!(lossFunction, params(ann), [(trainingDataset[1], trainingDataset[2])], ADAM(learningRate))
        trainLoss = lossFunction(ann(trainingDataset[1]), trainingDataset[2])
        valLoss = !isempty(validationDataset[1]) ? lossFunction(ann(validationDataset[1]), validationDataset[2]) : Inf
        
        if showText
            println("Epoch $epoch: Training Loss: $trainLoss, Validation Loss: $valLoss")
        end
        
        if valLoss < bestLoss
            bestLoss = valLoss
            bestModel = deepcopy(ann)
            epochNoUpgradeValidation = 0
        else
            epochNoUpgradeValidation += 1
            if epochNoUpgradeValidation >= maxEpochsVal
                break
            end
        end
    end
    
    # Optionally test the model if a test set is provided
    testLoss = !isempty(testDataset[1]) ? lossFunction(ann(testDataset[1]), testDataset[2]) : NaN
    (bestModel, trainLoss, valLoss, testLoss)
end

function trainClassANN(topology::AbstractArray{<:Int,1},
                       trainingDataset::Tuple{AbstractArray{<:Real,2}, AbstractArray{Bool,1}};
                       validationDataset::Tuple{AbstractArray{<:Real,2}, AbstractArray{Bool,1}} = 
                       (zeros(0, size(trainingDataset[1], 2)), falses(0)),
                       testDataset::Tuple{AbstractArray{<:Real,2}, AbstractArray{Bool,1}} = 
                       (zeros(0, size(trainingDataset[1], 2)), falses(0)),
                       transferFunctions::AbstractArray{<:Function,1} = fill(σ, length(topology)),
                       maxEpochs::Int = 1000, minLoss::Real = 0.0, learningRate::Real = 0.01,
                       maxEpochsVal::Int = 20, showText::Bool = false)
    
    # Preprocess to convert Bool array to 2D if necessary
    make2D(targets) = reshape(targets, :, 1)
    trainingDataset = (trainingDataset[1], make2D(trainingDataset[2]))
    validationDataset = (validationDataset[1], make2D(validationDataset[2]))
    testDataset = (testDataset[1], make2D(testDataset[2]))
    
    # Proceed with training
    return trainClassANN(topology, trainingDataset, validationDataset, testDataset, transferFunctions, maxEpochs, minLoss, learningRate, maxEpochsVal, showText)
end

function confusionMatrix(outputs::AbstractArray{Bool,2}, targets::AbstractArray{Bool,2}; weighted::Bool=true)
    numClasses = size(targets, 2)

    # Caso especial para una sola clase
    if numClasses == 1
        return confusionMatrix(outputs[:, 1], targets[:, 1])
    end

    # Inicialización de vectores para métricas
    metrics = Dict("recall" => zeros(numClasses),
                   "specificity" => zeros(numClasses),
                   "precision" => zeros(numClasses),
                   "NPV" => zeros(numClasses),
                   "F1" => zeros(numClasses))

    # Cálculo de métricas para cada clase que tiene instancias
    instancesPerClass = sum(targets, dims=1) |> vec
    for class in findall(instancesPerClass .> 0)
        for metric in keys(metrics)
            metrics[metric][class], _ = evaluateMetric(metric, outputs[:, class], targets[:, class])
        end
    end

    # Construcción de la matriz de confusión
    confMatrix = [sum(targets[:, i] .& outputs[:, j]) for i in 1:numClasses, j in 1:numClasses]

    # Aplicar ponderación o promedio según `weighted`
    if weighted
        weights = instancesPerClass / sum(instancesPerClass)
        weightedMetrics = map(x -> dot(weights, x), values(metrics))
        metrics = Dict(zip(keys(metrics), weightedMetrics))
    else
        activeClasses = sum(instancesPerClass .> 0)
        avgMetrics = map(x -> sum(x) / activeClasses, values(metrics))
        metrics = Dict(zip(keys(metrics), avgMetrics))
    end

    accuracyRate, errorRate = calculateRates(outputs, targets)

    return (accuracyRate, errorRate, metrics["recall"], metrics["specificity"], metrics["precision"], metrics["NPV"], metrics["F1"], confMatrix)
end



function evaluateMetric(metricName::String, outputs, targets)
    TP = sum(outputs .& targets)
    TN = sum(.!outputs .& .!targets)
    FP = sum(outputs .& .!targets)
    FN = sum(.!outputs .& targets)

    # Calcular diferentes métricas basadas en el nombre de la métrica
    metricValue = if metricName == "recall"
        TP + FN == 0 ? 0.0 : TP / (TP + FN)
    elseif metricName == "specificity"
        TN + FP == 0 ? 0.0 : TN / (TN + FP)
    elseif metricName == "precision"
        TP + FP == 0 ? 0.0 : TP / (TP + FP)
    elseif metricName == "NPV"
        TN + FN == 0 ? 0.0 : TN / (TN + FN)
    elseif metricName == "F1"
        2*TP + FP + FN == 0 ? 0.0 : 2*TP / (2*TP + FP + FN)
    else
        NaN
    end
    return metricValue, (TP, TN, FP, FN)
end

function calculateRates(outputs, targets)
    correctPredictions = sum(outputs .== targets)
    totalPredictions = length(outputs)
    accuracy = correctPredictions / totalPredictions
    errorRate = 1 - accuracy
    return accuracy, errorRate
end

function confusionMatrix(outputs::AbstractArray{Bool,1}, targets::AbstractArray{Bool,1})
    #Comprobamos que los vectores de salidas obtenidas y salidas deseadas sean de la misma longitud
    @assert(length(outputs)==length(targets));

    #Obtenemos los valores de VP, VN, FP, FN
    vp = sum(targets .& outputs);
    vn = sum(.!targets .& .!outputs);
    fp = sum(.!targets .& outputs);
    fn = sum(targets .& .!outputs);


    #Obtenemos la precisión y la tasa de error utilizando las funciones auxiliares
    acc = accuracy(outputs,targets);
    errorRate = 1. - acc;

    #Calculamos la sensibilidad, la especificidad, el valor predictivo positivo, el valor predictivo negativo y la F1-score
    recall = vp / (fn + vp);
    specificity = vn / (fp + vn);
    ppv = vp / (vp + fp);
    npv = vn / (vn + fn)
    f1 = (2 * recall * ppv) / (recall + ppv); 

    #Calculamos la matriz de confusión
    conf_matrix = Array{Int64,2}(undef, 2, 2);
    conf_matrix[1,1] = vn;
    conf_matrix[1,2] = fp;
    conf_matrix[2,1] = fn;
    conf_matrix[2,2] = vp;

    #Tenemos en cuenta varios casos particulares
    if (vn == length(targets))
        recall = 1.;
        ppv = 1.;
    elseif (vp == length(targets))
        specificity = 1.;
        npv = 1.;
    end

    recall = isnan(recall) ? 0. : recall;
    specificity = isnan(specificity) ? 0. : specificity;
    ppv = isnan(ppv) ? 0. : ppv;
    npv = isnan(npv) ? 0. : npv;

    f1 = (recall == ppv == 0.) ? 0. : 2 * (recall * ppv) / (recall + ppv);

    return (acc, errorRate, recall, specificity, ppv, npv, f1, conf_matrix);

end

function confusionMatrix(outputs::AbstractArray{<:Real,1}, targets::AbstractArray{Bool,1}; threshold::Real=0.5) 
    confusionMatrix(AbstractArray{Bool,1}(outputs.>=threshold),targets);
end

function confusionMatrix(outputs::AbstractArray{<:Real,2}, targets::AbstractArray{Bool,2}; weighted::Bool=true) 
    return confusionMatrix(classifyOutputs(outputs), targets, weighted = weighted);
end

function confusionMatrix(outputs::AbstractArray{<:Any,1}, targets::AbstractArray{<:Any,1}; weighted::Bool=true)
    @assert(all([in(output, unique(targets)) for output in outputs]))
    classes = unique(targets);
    confusionMatrix(oneHotEncoding(outputs, classes), oneHotEncoding(targets, classes); weighted=weighted);
end

function printConfusionMatrix(outputs::AbstractArray{Bool,1}, targets::AbstractArray{Bool,1})
    (acc, errorRate, recall, specificity, precision, NPV, F1, confMatrix) = confusionMatrix(outputs, targets)

    println("Confusion Matrix:")
    println(confMatrix)
    println("Accuracy: $acc")
    println("Error Rate: $errorRate")
    println("Recall (Sensitivity): $recall")
    println("Specificity: $specificity")
    println("Precision (Positive Predictive Value): $precision")
    println("Negative Predictive Value: $NPV")
    println("F1 Score: $F1")
    println(" ")

end

function printConfusionMatrix(outputs::AbstractArray{<:Real,2}, targets::AbstractArray{Bool,2}; weighted::Bool=true)
    printConfusionMatrix(classifyOutputs(outputs), targets; weighted=weighted);
end

function printConfusionMatrix(outputs::AbstractArray{Bool,2}, targets::AbstractArray{Bool,2}; weighted::Bool=true)
    acc, errorRate, recall, specificity, precision, NPV, F1, confMatrix = confusionMatrix(outputs, targets; weighted=weighted);
    
    println("Accuracy: ", acc);
    println("Error rate: ", errorRate);
    println("Recall: ", recall);
    println("Specificity: ", specificity);
    println("Precision: ", precision);
    println("Negative predictive value: ", NPV);
    println("F1-score: ", F1);
    return (acc, errorRate, recall, specificity, precision, NPV, F1, confMatrix);
end;




function printConfusionMatrix(outputs::AbstractArray{<:Real,1}, targets::AbstractArray{Bool,1}; threshold::Real=0.5)
    printConfusionMatrix(AbstractArray{Bool,1}(outputs.>=threshold),targets);    
end

function printConfusionMatrix(outputs::AbstractArray{<:Real,1}, targets::AbstractArray{<:Any,1}; threshold::Real=0.5, weighted::Bool=true)
    classifiedOutputs = outputs .>= threshold
    classifiedTargets = Bool.(targets)
    printConfusionMatrix(reshape(classifiedOutputs, :, 1), reshape(classifiedTargets, :, 1); weighted=weighted)
end

function crossvalidation(N::Int64, k::Int64)
    # Crear vector repetido de grupos y cortar al tamaño N
    group_vector = repeat(1:k, Int(ceil(N / k)))[:N]
    # Mezclar el vector de grupos para aleatorizar la asignación
    randomized_groups = shuffle(group_vector)
    return randomized_groups
end

function crossvalidation(targets::AbstractArray{Bool,2}, k::Int64)
    num_classes = size(targets, 2)
    num_samples = size(targets, 1)
    grouped_indices = Array{Int64,1}(undef, num_samples)
    
    # Asignar índices a grupos para cada clase de forma independiente
    for class_idx in 1:num_classes
        class_samples_indices = findall(x -> x, targets[:, class_idx])
        # Asegurarse de que hay suficientes muestras para cada clase
        @assert length(class_samples_indices) >= k "Not enough samples for class $class_idx"
        # Aplicar crossvalidation en los índices de las muestras de esta clase
        grouped_indices[class_samples_indices] = crossvalidation(length(class_samples_indices), k)
    end
    return grouped_indices
end

function crossvalidation(targets::AbstractArray{<:Any,1}, k::Int64)
    unique_classes = unique(targets)
    total_indices = length(targets)
    index_groups = Array{Int64,1}(undef, total_indices)
    
    # Procesar cada clase para asignar grupos de validación cruzada
    for each_class in unique_classes
        class_indices = findall(==(each_class), targets)
        num_class_samples = length(class_indices)
        # Verificar la viabilidad de k-fold
        @assert num_class_samples >= k "Not enough samples in class $each_class for $k-fold crossvalidation"
        # Aplicar crossvalidation a los índices de esta clase
        index_groups[class_indices] = crossvalidation(num_class_samples, k)
    end
    return index_groups
end

function crossvalidation(targets::AbstractArray{Bool,1}, k::Int64)
    # Calcular el número de verdaderos y falsos
    numTrue = sum(targets)
    numFalse = length(targets) - numTrue

    # Comprobar que hay suficientes muestras de cada clase para hacer la validación cruzada
    @assert numTrue >= k && numFalse >= k "Not enough samples of either True or False class for $k-fold crossvalidation"

    # Crear vector de índices para resultados True y False
    trueIndices = findall(x -> x, targets)
    falseIndices = findall(x -> !x, targets)

    # Aplicar crossvalidation a los índices separadamente
    randomizedTrueIndices = shuffle(trueIndices)
    randomizedFalseIndices = shuffle(falseIndices)

    # Asignar grupos
    groups = Array{Int64}(undef, length(targets))
    groupNumbersTrue = repeat(1:k, Int(ceil(numTrue / k)))[1:numTrue]
    groupNumbersFalse = repeat(1:k, Int(ceil(numFalse / k)))[1:numFalse]

    shuffle!(groupNumbersTrue)
    shuffle!(groupNumbersFalse)

    groups[randomizedTrueIndices] = groupNumbersTrue
    groups[randomizedFalseIndices] = groupNumbersFalse

    return groups
end

function modelCrossValidation(modelType::Symbol, modelHyperparameters::Dict, inputs::AbstractArray{<:Real,2}, targets::AbstractArray{<:Any,1}, crossValidationIndices::Array{Int64,1})
    # Asegurar coherencia entre el número de entradas y el número de etiquetas
    @assert size(inputs, 1) == length(targets) "Mismatch between number of inputs and number of targets."

    # Identificar las clases únicas para posible one-hot encoding
    uniqueClasses = unique(targets)
    
    # Realizar one-hot encoding si se entrena una red neuronal
    encodedTargets = modelType == :ANN ? oneHotEncoding(targets, uniqueClasses) : targets

    # Definir las métricas a calcular
    accuracies = Float64[]
    f1Scores = Float64[]

    # Dividir el conjunto de datos según los índices de validación cruzada
    numFolds = maximum(crossValidationIndices)
    for fold in 1:numFolds
        # Preparar subconjuntos de entrenamiento y prueba
        trainIdx = crossValidationIndices .!= fold
        testIdx = crossValidationIndices .== fold

        trainInputs, trainTargets = inputs[trainIdx, :], encodedTargets[trainIdx]
        testInputs, testTargets = inputs[testIdx, :], encodedTargets[testIdx]

        # Configurar y entrenar el modelo
        model = chooseModel(modelType, modelHyperparameters)
        fit!(model, trainInputs, trainTargets)

        # Evaluar el modelo en el conjunto de prueba
        predicted = predict(model, testInputs)
        accuracy, f1 = evaluateModel(predicted, testTargets)

        # Almacenar resultados de cada pliegue
        push!(accuracies, accuracy)
        push!(f1Scores, f1)
    end

    # Informe de resultados
    avgAccuracy = mean(accuracies)
    stdAccuracy = std(accuracies)
    avgF1 = mean(f1Scores)
    stdF1 = std(f1Scores)

    println("Cross-validation results -- Accuracy: Mean = $avgAccuracy, SD = $stdAccuracy; F1-Score: Mean = $avgF1, SD = $stdF1")

    return avgAccuracy, stdAccuracy, avgF1, stdF1
end

# Funciones auxiliares para encapsular la selección del modelo y evaluación
function chooseModel(modelType::Symbol, hyperparams::Dict)
    if modelType == :SVM
        return SVC(; hyperparams...)
    elseif modelType == :DecisionTree
        return DecisionTreeClassifier(; hyperparams...)
    elseif modelType == :kNN
        return KNeighborsClassifier(; hyperparams...)
    elseif modelType == :ANN
        return buildClassANN(hyperparams["inputSize"], hyperparams["topology"], hyperparams["outputSize"], hyperparams["transferFunctions"])
    else
        error("Unsupported model type: $modelType")
    end
end

function evaluateModel(predictions, actual)
    # Implementación simple de cálculo de precisión y F1
    tp = sum(predictions .& actual)
    fp = sum(predictions .& .!actual)
    fn = sum(.!predictions .& actual)
    tn = sum(.!predictions .& .!actual)

    accuracy = (tp + tn) / length(actual)
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1 = 2 * precision * recall / (precision + recall)

    return accuracy, f1
end
