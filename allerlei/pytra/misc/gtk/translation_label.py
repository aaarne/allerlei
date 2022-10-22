#!/usr/bin/env python
import gtk


class TranslationLabel(gtk.HBox):
    def __init__(self,mm=True):
        super(TranslationLabel, self).__init__()
        self.__xlabel = gtk.Label('x')
        self.__ylabel = gtk.Label('y')
        self.__zlabel = gtk.Label('z')
        self.pack_start(self.__xlabel)
        self.pack_start(self.__ylabel)
        self.pack_start(self.__zlabel)
        self.__mm = mm

    @staticmethod
    def float_to_string(f, mm=True):
        if mm:
            return '%.2f' % (1000*f,)
        return '%.5f' % (f,)

    def set_translation(self, translation):
        t = translation.getTranslationVector()
        self.__xlabel.set_text(TranslationLabel.float_to_string(t[0], self.__mm))
        self.__ylabel.set_text(TranslationLabel.float_to_string(t[1], self.__mm))
        self.__zlabel.set_text(TranslationLabel.float_to_string(t[2], self.__mm))
