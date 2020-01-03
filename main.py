from rnn import main as rnn_main


FLAG = 'RNN'


def main():
	if FLAG == 'RNN':
		rnn_main(hidden_dim=32, batch_size=1)



if __name__ == '__main__':
	main()
