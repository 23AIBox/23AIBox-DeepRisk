from tensorflow.python.framework import dtypes
from tensorflow.python.keras import layers
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.engine.input_spec import InputSpec
from tensorflow.python.keras.utils import tf_utils
from tensorflow.python.util.tf_export import keras_export


class PartialDense(layers.Dense):

	def __init__(self, adjacent_table, **kwargs):

		kwargs['units'] = -1

		super(PartialDense, self).__init__(**kwargs)

		self.adjacent_table = adjacent_table

	@tf_utils.shape_type_conversion
	def build(self, input_shape):
		dtype = dtypes.as_dtype(self.dtype or K.floatx())
		if not (dtype.is_floating or dtype.is_complex):
			raise TypeError('Unable to build `Dense` layer with non-floating point '
							'dtype %s' % (dtype,))

		input_dim = input_shape[1]

		rows, cols = zip(*self.adjacent_table)

		assert input_dim == max(rows) + 1

		if input_dim is None:
			raise ValueError('Axis 2 of input should be fully-defined. '
							 'Found shape:', input_shape)

		self.units = max(cols) + 1

		self.kernel_indices = K.constant(list(zip(cols, rows)), dtype='int64')
		self.kernel_shape = K.constant([self.units, input_dim], dtype='int64')

		self.input_spec = InputSpec(min_ndim=2, axes={-1: input_dim})

		self.kernel = self.add_weight(
			'kernel',
			shape=[self.kernel_indices.shape[0]],
			initializer=self.kernel_initializer,
			regularizer=self.kernel_regularizer,
			constraint=self.kernel_constraint,
			dtype=self.dtype,
			trainable=True)

		if self.use_bias:
			self.bias = self.add_weight(
				'bias',
				shape=[self.units,],
				initializer=self.bias_initializer,
				regularizer=self.bias_regularizer,
				constraint=self.bias_constraint,
				dtype=self.dtype,
				trainable=True)
		else:
			self.bias = None

		self.built = True

	def call(self, inputs):

		output = K.sparse_ops.sparse_tensor_dense_mat_mul(self.kernel_indices, self.kernel, self.kernel_shape, inputs, adjoint_b=True)

		output = K.transpose(output)

		if self.use_bias:
			output = K.bias_add(output, self.bias)

		output = self.activation(output)
		return output


class PartialConnected1D(layers.Dense):

	def __init__(self, adjacent_table, **kwargs):

		# kwargs['kernel_size'] = 1

		super(PartialConnected1D, self).__init__(**kwargs)

		self.adjacent_table = adjacent_table

	@tf_utils.shape_type_conversion
	def build(self, input_shape):
		dtype = dtypes.as_dtype(self.dtype or K.floatx())
		if not (dtype.is_floating or dtype.is_complex):
			raise TypeError('Unable to build `Dense` layer with non-floating point '
							'dtype %s' % (dtype,))

		input_length, input_dim = input_shape[1], input_shape[2]

		if input_dim is None:
			raise ValueError('Axis 2 of input should be fully-defined. '
							 'Found shape:', input_shape)

		rows, cols = zip(*self.adjacent_table)

		assert input_length == max(rows) + 1

		self.output_length = max(cols) + 1

		kernel_indices = list()
		for r, c in self.adjacent_table:
			for i in range(input_dim):
				for j in range(self.units):
					kernel_indices.append((c * self.units + j, r * input_dim + i))
		self.kernel_indices = K.constant(kernel_indices, dtype='int64')
		self.kernel_shape = K.constant([self.output_length * self.units, input_length * input_dim], dtype='int64')

		self.kernel = self.add_weight(
			'kernel',
			shape=[self.kernel_indices.shape[0]],
			initializer=self.kernel_initializer,
			regularizer=self.kernel_regularizer,
			constraint=self.kernel_constraint,
			dtype=self.dtype,
			trainable=True)

		if self.use_bias:
			self.bias = self.add_weight(
				'bias',
				shape=[self.output_length, self.units],
				initializer=self.bias_initializer,
				regularizer=self.bias_regularizer,
				constraint=self.bias_constraint,
				dtype=self.dtype,
				trainable=True)
		else:
			self.bias = None

		self.input_spec = InputSpec(ndim=3, axes={-1: input_dim})
		self.built = True

	def call(self, inputs):

		batch_size = K.shape(inputs)[0]

		inputs = K.reshape(inputs, (batch_size, -1))

		output = K.sparse_ops.sparse_tensor_dense_mat_mul(self.kernel_indices, self.kernel, self.kernel_shape, inputs, adjoint_b=True)

		output = K.transpose(output)
		output = K.reshape(output, (batch_size, self.output_length, self.units))

		if self.use_bias:
			output = K.bias_add(output, self.bias)

		output = self.activation(output)
		return output


if __name__ == "__main__":
	
	mask = [[0,0],[1,1],[2,1]]

	from tensorflow.keras import Sequential

	model = Sequential([
		layers.Input(shape=(3,)),
		PartialDense(mask)
	])

	model.compile()

	model.summary()
