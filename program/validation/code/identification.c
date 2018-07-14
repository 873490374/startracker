/****************************************************************************
 * This file is in the public domain.                                       *
 * Remove this comment when you edit the file and give it a proper          *
 * copyright and license.                                                   *
 ****************************************************************************/

/***
 * Prepares the data from the data file for final use.
 * This function should not run expensive preprocessing steps.
 * It should just put the data in a structure to be
 * usable by the star_id function.
 * 
 * @param data The data loaded from the _data_ file.
 * @param data_size The size of the data loaded from the file.
 * @return The prepared data.
 */
void* prepare(void* data, long data_size)
{
	return data;
}

/***
 * Determines the HIP numbers of stars in a scene.
 * 
 * @param spikes The data of the spikes in the scene. x, y, and magnitude interleaved.
 * @param result The output array that should be filled with the HIP numbers.
 * @param length The number of spikes in the scene.
 * @param data The data prepared by the prepare function.
 */
void star_id(double spikes[], int result[], size_t length, void* data)
{
	for(size_t i = 0; i < length; i++)
	{
		result[i] = -1;
	}
}
