#include <unordered_map>
#include <vector>

namespace trie {
    template <typename T>
    class tnode {
    public:
        explicit tnode(T v) : mapped_value(v) {};
        void addChild(tnode* child, int key);
        tnode<T>* getChild(int key);
        T& get() { return mapped_value; };
        void update(T val);
        std::unordered_map<int,tnode<T>*> children;
    private:
        T mapped_value;
    };

    template <typename T>
    void tnode<T>::addChild(tnode* child, int key) {
        children[key] = child;
    }

    template <typename T>
    tnode<T>* tnode<T>::getChild(int key) {
        const auto child = children.find(key);
        if (child==children.end())
            return nullptr;
        return child->second;
    }

    template <typename T>
    void tnode<T>::update(T val) {
        mapped_value = val;
    }
} // namespace trie
