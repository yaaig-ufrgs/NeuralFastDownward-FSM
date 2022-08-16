#ifndef TRIE_TRIE_NODE_H
#define TRIE_TRIE_NODE_H

#include <unordered_map>
#include <vector>

namespace trie {
template <typename T>
class tnode {
public:
    explicit tnode(T v, tnode<T>* p, int ascii, bool eow = false);
    void addChild(tnode* child, int key);
    std::vector<int> getKey();
    tnode<T>* getChild(int key);
    T& get();
    void update(T val);
    void markEnd(std::vector<int>);
    bool isEnd();
    tnode<T>* getParent();
    int getParentIndex();
private:
    T mapped_value;
    // int p_index;
    // bool isEndOfWord;
    // tnode<T>* parent;
    std::unordered_map<int,tnode<T>*> children;
    // std::vector<int> key;
};

template <typename T>
tnode<T>::tnode(T val, tnode<T>* /*p*/, int /*ascii*/, bool /*eow*/) {
    this->mapped_value = val;
    // this->isEndOfWord = eow;
    // this->p_index = ascii;
    // this->key = {};
    // this->parent = p;
}

template <typename T>
void tnode<T>::addChild(tnode* child, int key) {
    this->children[key] = child;
}

template <typename T>
std::vector<int> tnode<T>::getKey() {
  exit(1516);// WTF?
    return {};
    // return this->key;
}

template <typename T>
tnode<T>* tnode<T>::getChild(int key) {
  const auto child = children.find(key);
  if (child==children.end())
    return nullptr;
  return child->second;
}

template <typename T>
T& tnode<T>::get() {
    return this->mapped_value;
}

template <typename T>
void tnode<T>::update(T val) {
    this->mapped_value = val;
}

template <typename T>
void tnode<T>::markEnd(std::vector<int> /*key*/) {
    // this->key = key;
    // this->isEndOfWord = true;
}

template <typename T>
bool tnode<T>::isEnd() {
    return true;
    // return this->isEndOfWord;
}

template <typename T>
tnode<T>* tnode<T>::getParent() {
  exit(1516); // WTF?
    return nullptr;
    // return this->parent;
}

template <typename T> int tnode<T>::getParentIndex() {
  exit(1516); // WTF?
    return -1;
    // return this->p_index;
}
} // namespace trie

#endif
