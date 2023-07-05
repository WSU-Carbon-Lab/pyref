#include <stdio.h>

int findMaxIndex2D(unsigned int *arr, int rows, int cols)
{
    int maxIndex = 0;
    unsigned int maxValue = arr[0];
    int i, j;

    for (i = 0; i < rows; i++)
    {
        for (j = 0; j < cols; j++)
        {
            unsigned int currentValue = arr[i * cols + j];
            if (currentValue > maxValue)
            {
                maxValue = currentValue;
                maxIndex = i * cols + j;
            }
        }
    }

    return maxIndex;
}