#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>

class KNeighborsClassifier {
public:
    KNeighborsClassifier(int k) : k(k) {}

    void fit(const std::vector<int>& age, const std::vector<int>& high) {
        this->age = age;
        this->high = high;
    }

    int predict(int age_test) {
        std::vector<std::pair<int, int> > distances;

        for (size_t i = 0; i < age.size(); ++i) {
            int dist = std::abs(age[i] - age_test);
            distances.emplace_back(dist, high[i]);
        }

        std::sort(distances.begin(), distances.end());

        int sum = 0;
        for (int i = 0; i < k; ++i) {
            sum += distances[i].second;
        }

        return sum / k;
    }

private:
    std::vector<int> age;
    std::vector<int> high;
    int k;
};

int main() {
    std::vector<int> age = {1, 4, 7, 9, 10, 12, 15, 18, 20, 24, 30, 40, 50};
    std::vector<int> high = {30, 50, 80, 90, 100, 120, 140, 180, 185, 178, 188, 174, 168};

    int age_test, k;
    std::cout << "请输入一个想要预测的年龄值: ";
    std::cin >> age_test;
    std::cout << "输入一个想要预测的K 值: ";
    std::cin >> k;

    KNeighborsClassifier knn(k);
    knn.fit(age, high);
    int high_pred = knn.predict(age_test);

    std::cout << "预测的身高为: " << high_pred << std::endl;

    return 0;
}
4
