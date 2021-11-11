#!/usr/bin/env python
import gtk


class RotationLabel(gtk.Table):
    def __init__(self):
        super(RotationLabel, self).__init__(3,3)
        self.__r_labels = {}
        for x in ['11','12','13','21','22','23','31','32','33']:
            self.__r_labels[x] = gtk.Label(x)

        for key in self.__r_labels.keys():
            self.attach(self.__r_labels[key], int(key[1])-1, int(key[1]), int(key[0])-1, int(key[0]))

    @staticmethod
    def float_to_string(f):
        return '%.4f' % (f,)

    def set_rotation(self, rotation):
        rotmat = rotation.getRotationMatrix()
        for (nrow, row) in enumerate(rotmat):
            for (ncol, col) in enumerate(row):
                self.__r_labels[str(nrow+1) + str(ncol+1)].set_text(RotationLabel.float_to_string(col))
