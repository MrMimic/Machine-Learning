
#!/usr/bin/env python3




import numpy as np
import pandas as pd
from collections import Counter
from matplotlib import pyplot as plt
import matplotlib.patches as mpatches




class Survival(object):


	def __init__(self):
		print()
		self.training_data = pd.read_csv('./train.csv')
		self.predict_data = pd.read_csv('./test.csv')
		self.plot_stats = False
		self.l_rate = 0.1
		self.bias = 0
		self.epochs = 1000


	def explore_data(self):
		""" Explore data from Kaggle """
		print('Data available:\t{}\n'.format(self.training_data.columns.values.tolist()))

		# Age
		age_data = self.training_data['Age'].value_counts(sort=False).sort_index()
		# Sex
		sex_data = self.training_data['Sex'].value_counts()
		# Proportion of deads
		dead_data = self.training_data['Survived'].value_counts()
		# Ticket fare VS Pclass
		fare_pclass = self.training_data[['Fare', 'Pclass']]
		price_data = fare_pclass.groupby('Pclass').mean()

		# PLOT
		colors = ['olive', 'blue']

		plt.subplot(221)
		plt.hist(self.training_data['Age'].dropna(), color=colors[np.random.randint(low=0, high=len(colors))])
		plt.xlabel('Age (years)')
		plt.ylabel('Frequency')

		plt.subplot(222)
		plt.pie(sex_data.values, colors = colors)
		red_patch = mpatches.Patch(color=colors[0], label=sex_data.index[0])
		blue_patch = mpatches.Patch(color=colors[1], label=sex_data.index[1])
		plt.legend(handles=[red_patch, blue_patch], loc='lower right')

		plt.subplot(223)
		plt.pie(dead_data.values, colors = colors)
		red_patch = mpatches.Patch(color=colors[0], label='Died' if dead_data.index[0] == 0 else 'Survived')
		blue_patch = mpatches.Patch(color=colors[1], label='Died' if dead_data.index[1] == 0 else 'Survived')
		plt.legend(handles=[red_patch, blue_patch], loc='lower left')

		plt.subplot(224)
		plt.bar(price_data.index, price_data.values, color=colors[np.random.randint(low=0, high=len(colors))])
		plt.xlabel('Ticket class')
		plt.ylabel('Ticket price ($)')
		plt.xticks(range(1, len(price_data.index)+1))

		plt.suptitle('Titanic dataset visualization')
		
		if self.plot_stats is True:
			plt.show()
		plt.close(1)


	def prepare_data(self):

		output_data_pd = self.training_data[['Survived']]
		output_data = output_data_pd['Survived'].tolist()

		input_data_pd = self.training_data[['Pclass', 'Sex', 'Age', 'Fare']]
		input_data_nottransformed = pd.DataFrame(input_data_pd).values.tolist()
		
		# Let's transform raw data into vector of integers
		input_data = []
		for raw_data in input_data_nottransformed:
			line = []
			# Ticket class
			line.append(raw_data[0])
			# Sex
			if raw_data[1] == 'male':
				line.append(1)
			else:
				line.append(0)
			# Age
			try:
				line.append(int(raw_data[2]))
			except ValueError:
				line.append(0)
			# Fare
			line.append(int(raw_data[3]))
			input_data.append(line)



		return input_data, output_data


	def predict(self, row, weights):
		""" Predict output class from a list of values and a vector of weights """

		activation = 0
		for i in range(len(row) - 1):
			activation += (weights[i] * row[i])
		activation += self.bias
		# Return class
		if activation >= 0.0:
			return 1
		else:
			return 0


	def train_weights(self, input_data, output_data):
		""" Train our perceptron """

		# Let's initiate weights with small values
		weights = list(np.random.uniform(low = 0, high = 0.1, size = 4))
		global_errors = []
		for epoch in range(self.epochs):
			# Keep track of errors for plotting
			epoch_errors = 0.0
			for row, expected in zip(input_data, output_data):

				prediction = self.predict(row, weights)
				error = expected - prediction
				# We keep a track of global errors for plotting
				global_errors.append(abs(error))
				epoch_errors += abs(error)
				# If predicted and expected are different, update weights and bias
				if expected != prediction:
					self.bias = self.bias + self.l_rate * error
					for i in range(len(row)-1):
						weights[i] = weights[i] + self.l_rate * error * row[i]
			if epoch % 100 == 0:
				print('epoch: {}, lrate: {}, errors: {}'.format(epoch, self.l_rate, epoch_errors))
		print('\nWeights trained: {}\n'.format(weights))

		return weights


	def get_input_data(self):

		predict_data_pd = self.predict_data[['Pclass', 'Sex', 'Age', 'Fare']]
		predict_data_nottransformed = pd.DataFrame(predict_data_pd).values.tolist()
		predict_data = []
		for raw_data in predict_data_nottransformed:
			line = []
			# Ticket class
			line.append(raw_data[0])
			# Sex
			if raw_data[1] == 'male':
				line.append(1)
			else:
				line.append(0)
			# Age
			try:
				line.append(int(raw_data[2]))
			except ValueError:
				line.append(0)
			# Fare
			try:
				line.append(int(raw_data[3]))
			except ValueError:
				line.append(0)

			predict_data.append(line)

		return predict_data


	def summarize_predictions(self, predictions):


		predictions = pd.DataFrame(predictions, columns=['Class', 'Sex', 'Age', 'Fare', 'Statut'])

		# Statistics
		data_survived = predictions.loc[predictions['Statut'] == 1]
		data_dead = predictions.loc[predictions['Statut'] == 0]
		print('DEAD: {}/{}'.format(len(data_dead), len(predictions)))
		print('ALIVE: {}/{}'.format(len(data_survived), len(predictions)))

		# Sex __ survived
		plt.subplot(241).set_title('Sex __ survived')
		plt.pie(data_survived['Sex'].groupby(data_survived['Sex']).count(), colors=['blue', 'cyan'])
		blue_patch = mpatches.Patch(color='blue', label=data_survived['Sex'].groupby(data_survived['Sex']).count().index[0])
		cyan_patch = mpatches.Patch(color='cyan', label=data_survived['Sex'].groupby(data_survived['Sex']).count().index[1])
		plt.legend(handles=[blue_patch, cyan_patch], loc='lower right')
		# Class __ survived
		plt.subplot(242).set_title('Class __ survived')
		plt.bar(
			data_survived['Class'].groupby(data_survived['Class']).count().index, 
			data_survived['Class'].groupby(data_survived['Class']).count().values, 
			color='blue')
		plt.xticks(range(1, 4))
		# Age __ survived
		plt.subplot(243).set_title('Age __ survived')
		plt.hist(data_survived['Age'].dropna(), color='cyan')
		plt.xlabel('Age (years)')
		plt.ylabel('Frequency')
		plt.xticks(range(0, 90, 10))
		# Fare __ survived
		plt.subplot(244).set_title('Fare __ survived')
		plt.hist(data_survived['Fare'].dropna(), color='blue')
		plt.xlabel('Fare ($)')
		plt.ylabel('Frequency')
		plt.xticks(range(0, 600, 50))
		plt.xlim(0, 150)

		# Sex __ dead
		plt.subplot(245).set_title('Sex __ dead')
		plt.pie(data_dead['Sex'].groupby(data_dead['Sex']).count(), colors=['red', 'black'])
		red_patch = mpatches.Patch(color='red', label=data_dead['Sex'].groupby(data_dead['Sex']).count().index[0])
		#~ black_patch = mpatches.Patch(color='black', label=data_dead['Sex'].groupby(data_dead['Sex']).count().index[1])
		plt.legend(handles=[red_patch], loc='lower right')
		# Class __ dead
		plt.subplot(246).set_title('Class __ dead')
		plt.bar(
			data_dead['Class'].groupby(data_dead['Class']).count().index, 
			data_dead['Class'].groupby(data_dead['Class']).count().values, 
			color='brown')
		plt.xticks(range(1, 4))
		# Age __ dead
		plt.subplot(247).set_title('Age __ dead')
		plt.hist(data_dead['Age'].dropna(), color='red')
		plt.xlabel('Age (years)')
		plt.ylabel('Frequency')
		plt.xticks(range(0, 90, 10))
		# Fare __ dead
		plt.subplot(248).set_title('Fare __ dead')
		plt.hist(data_dead['Fare'].dropna(), color='brown')
		plt.xlabel('Fare ($)')
		plt.ylabel('Frequency')
		plt.xticks(range(0, 600, 50))
		plt.xlim(0, 150)


		plt.show()



if __name__ == '__main__':

	survival = Survival()

	survival.explore_data()
	input_data, output_data = survival.prepare_data()
	weights = survival.train_weights(input_data=input_data, output_data=output_data)
	predict_data = survival.get_input_data()
	predictions = []
	for passenger in predict_data:
		statut = survival.predict(row=passenger, weights=weights)
		passenger_translated = []
		passenger_translated.append(passenger[0])
		passenger_translated.append('Men' if passenger[1] == 1 else 'Women')
		passenger_translated.append(passenger[2])
		passenger_translated.append(passenger[3])
		passenger_translated.append(statut)
		predictions.append(passenger_translated)
	survival.summarize_predictions(predictions=predictions)



