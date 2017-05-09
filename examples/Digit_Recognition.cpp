#include <iostream>
#include <cstdlib>
#include <vector>
#include <cmath>
#include <time.h>
#include <SFML/Graphics.hpp>
#include <chai.h>
#include <loadmnist.h>

unsigned int MaxElement(std::vector<float> input);
std::vector<std::vector<float> > LoadMNIST();

unsigned int MaxElement(std::vector<float> input)
{
  if (input.size() == 0)
    return -1;

  float max = input[0];
  int index = 0;
  for (unsigned int i = 1; i < input.size(); i++)
    if (input[i] > max)
    {
      max = input[i];
      index = i;
    }

  return index;
}

int main()
{
    srand(time(NULL));
    std::vector<std::vector<float> > training_data = LoadMNIST();

    //create the labels
    std::vector<std::vector<float> > labels;
    for (int i = 0; i < 10; i++)
    {
        std::vector<float> label;
        label.resize(10);
        label[i] = 1.0f;
        labels.push_back(label);
    }

    ChaiModel MNISTModel;
    MNISTModel.AddFCL(784, 10);
    MNISTModel.AddSoftmax(10);
    const unsigned int INPUT_SIZE = 784;
    const unsigned int NUM_CLASSES = 10;
    const unsigned int OUTPUT_SIZE = 10;
    const unsigned int NUM_EPOCHS = 10;
    const float LEARNING_RATE = 0.1f;
    const unsigned int NUM_EXAMPLES = training_data[0].size() / INPUT_SIZE;

    for (unsigned int epoch = 0; epoch < NUM_EPOCHS; epoch++)
      for (unsigned int i = 0; i < NUM_EXAMPLES; i++)
        for (unsigned int j = 0; j < NUM_CLASSES; j++)
        {
          std::vector<float> input;
          input.reserve(INPUT_SIZE);
          for (int k = i * INPUT_SIZE; k < i * INPUT_SIZE + INPUT_SIZE; k++)
            input.push_back(training_data[j][k]);

          float loss = MNISTModel.Train(input, labels[j], LEARNING_RATE / pow(10, epoch / 10.0f));

          //train the model
          if (i % 100 == 0)
          std::cout << "Epoch " << epoch + 1 << ", loss: " << loss << std::endl;
        }

  std::cout << "Testing trained model... " << std::endl;

  unsigned int num_tests = 0;
  unsigned int num_correct = 0;
  for (unsigned int i = 0; i < NUM_EXAMPLES; i++)
  {
    for (unsigned int j = 0; j < NUM_CLASSES; j++)
    {
      std::vector<float> input;
      input.reserve(INPUT_SIZE);
      for (int k = i * INPUT_SIZE; k < i * INPUT_SIZE + INPUT_SIZE; k++)
        input.push_back(training_data[j][k]);

      num_tests++;

      if (MaxElement(MNISTModel.Evaluate(input)) == j)
        num_correct++;
    }
  }

  std::cout << "Accuracy: " << (float)num_correct / num_tests * 100.0f << "%" << std::endl;

  // Create the main window
  sf::RenderWindow window(sf::VideoMode(280, 280), "Handwriting Recognition");
  sf::VertexArray pointmap(sf::Points, 280 * 280);
  for (int i = 0; i < 280; i++)
      for (int j = 0; j < 280; j++)
      {
          pointmap[i * 280 + j].position.x = j;
          pointmap[i * 280 + j].position.y = i;
          pointmap[i * 280 + j].color = sf::Color::Black;
      }

  // Start the game loop
  while (window.isOpen())
  {
      // Process events
      sf::Event event;
      while (window.pollEvent(event))
      {
          // Close window: exit
          if (event.type == sf::Event::Closed) {
              window.close();
          }

          // Escape pressed: exit
          if (event.type == sf::Event::KeyPressed && event.key.code == sf::Keyboard::Escape) {
              window.close();
          }
      }

      //zoom into area that is left clicked
      if (sf::Mouse::isButtonPressed(sf::Mouse::Left))
      {
          sf::Vector2i position = sf::Mouse::getPosition(window);
          if (position.x + position.y * 280 < 280*280 && position.x + position.y * 280 >= 0)
          {
              for (int i = -16; i < 17; i++)
              {
                  for (int j = -16; j < 17; j++)
                  {
                      if (position.x + i + (position.y + j) * 280 < 280*280 && position.x + i + (position.y + j) * 280 >= 0)
                      {
                          float distance_squared = i * i + j * j + 1;
                          sf::Color color(255 / distance_squared, 255 / distance_squared, 255 / distance_squared);
                          pointmap[position.x + i + (position.y + j) * 280].position.x = position.x + i;
                          pointmap[position.x + i + (position.y + j) * 280].position.y = position.y + j;
                          pointmap[position.x + i + (position.y + j) * 280].color += color;
                          pointmap[position.x + i + (position.y + j) * 280].color += color;
                          pointmap[position.x + i + (position.y + j) * 280].color += color;
                          pointmap[position.x + i + (position.y + j) * 280].color += color;
                          pointmap[position.x + i + (position.y + j) * 280].color += color;
                      }
                  }
              }
          }
      }

      if (sf::Keyboard::isKeyPressed(sf::Keyboard::Key::C))
      {
          for (int i = 0; i < 280*280; i++)
              pointmap[i].color = sf::Color::Black;
      }

      if (sf::Keyboard::isKeyPressed(sf::Keyboard::Key::Return))
      {
          std::vector<float> input;
          input.reserve(784);
          for (int k = 0; k < 28; k++)
          {
              for (int l = 0; l < 28; l++)
              {
                  float average = 0.0f;
                  for (int i = 0; i < 10; i++)
                      for (int j = 0; j < 10; j++)
                      {
                          float temp_average = 0.0f;
                          temp_average += pointmap[k * 10 * 280 + l * 10 + i * 280 + j].color.r;
                          temp_average += pointmap[k * 10 * 280 + l * 10 + i * 280 + j].color.g;
                          temp_average += pointmap[k * 10 * 280 + l * 10 + i * 280 + j].color.b;
                          temp_average /= 3.0f;
                          average += temp_average;
                      }
                  average /= 100.0f;
                  average /= 255.0f; //normalize
                  input.push_back(average);
              }
          }
          std::vector<float> output = MNISTModel.Evaluate(input);
          int prediction = MaxElement(output);
          std::cout << "This number is predicted to be a: " << prediction << std::endl;
      }

      // Clear screen
      window.clear();
      window.draw(pointmap);
      // Update the window
      window.display();
  }

  return EXIT_SUCCESS;
}
