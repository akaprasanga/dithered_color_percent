

class FunctionsFromDLL:


    def dither_image(self,filename, number, dir_path):
        import os
        import clr
        if not os.path.exists('A_DITHERED'):
            os.makedirs('A_DITHERED')
        saving_path = dir_path + '/A_DITHERED/'
        utilities_path = dir_path + '/Utilities.dll'
        clr.AddReference(utilities_path)
        from Utilities import ImageProcessing
        obj = ImageProcessing()
        obj.Dithering(saving_path, filename, number)
        return saving_path

    def reduce_color(self,filename, number, dir_path):
        import os
        import clr
        if not os.path.exists('A_REDUCE_COLOR'):
            os.makedirs('A_REDUCE_COLOR')
        saving_path = dir_path + '/A_REDUCE_COLOR/'
        utilities_path = dir_path + '/Utilities.dll'
        clr.AddReference(utilities_path)
        from Utilities import ImageProcessing
        obj = ImageProcessing()
        saved_path = obj.AutoReduceColor(saving_path, filename, number)
        return saved_path

    def replace_mean_color(self,original_filename, index_filename, dir_path):
        import os
        import clr
        if not os.path.exists('A_OUTPUT'):
            os.makedirs('A_OUTPUT')
        saving_path = dir_path + '/A_OUTPUT/'
        utilities_path = dir_path + '/Utilities.dll'
        clr.AddReference(utilities_path)
        from Utilities import ImageProcessing
        obj = ImageProcessing()
        # orig_file = cv2.imread(original_filename)
        # index_file = cv2.imread(index_filename)
        # print('Before going to replace color ', orig_file.shape, index_file.shape)
        saved_path = obj.ReplacewithMean(saving_path, original_filename, index_filename)
        return saved_path

