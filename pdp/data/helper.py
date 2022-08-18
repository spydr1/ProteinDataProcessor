from dataclasses import dataclass
import abc
import tensorflow as tf


class AbstractDataclass(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def get_example(self) -> tf.train.Example:
        """
        Convert feature to the serialized feature.
        """
        raise NotImplementedError()


class AbstractProcessor(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def __next__(self):
        """
        Get next data
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def __iter__(self):
        raise NotImplementedError()
