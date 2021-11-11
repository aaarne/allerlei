#!/usr/bin/env python
import gtk


class TransformationLabel(gtk.Table):
    def __init__(self):
        super(TransformationLabel, self).__init__(4,4)
        self.__r_labels = {}
        for x in ['11','12','13', '14','21','22','23','24','31','32','33','34','41','42','43','44']:
            self.__r_labels[x] = gtk.Label(x)

        for key in self.__r_labels.keys():
            self.attach(self.__r_labels[key], int(key[1])-1, int(key[1]), int(key[0])-1, int(key[0]))

    @staticmethod
    def float_to_string(f):
        return '%.4f' % (f,)

    def set_transformation(self, transformation):
        transmat = transformation.getTransformationMatrix()
        for (nrow, row) in enumerate(transmat):
            for (ncol, col) in enumerate(row):
                self.__r_labels[str(nrow+1) + str(ncol+1)].set_text(TransformationLabel.float_to_string(col))
