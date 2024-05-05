# AAentrega
Archivos para la entrega de la práctica de AA: Identificación de notas musicales. Curso 2023/2024.

Las aproximaciones se ejecutan desde el archivo con su nombre.

Los parámetros de los sistemas (RNA, SVM, árbol de decisión, kNN) se cambian donde se definen.

La función procesar_archivos de las aproximaciones 2, 3, 4 y 5 se ejecuta solo la primera vez, y luego debe comentarse con un símbolo #. Esta es la función que procesa la entrada y guarda los valores en un .txt, para ahorrar tiempo, y solo se necesita ejecutar la primera vez.

El tamaño de los vectores de entrada (input_length) y salida (output_length), y el número de muestras por ventana (muestras_input), aunque no debería cambiarse, están arriba del todo junto la definición del nómero de k-fold.

Al final del código está la mencionada función procesar_archivos y la definición de la semilla, junto con la llamada al modelCrossValidation en las aproximaciones donde se utiliza. Ejecutamos modelCrossValidation 4 veces con cada arquitectura para hacer la boxplot de los resultados.
