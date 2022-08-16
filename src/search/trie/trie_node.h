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
    tnode<T>* getChild(int key);
    T& get();
    void update(T val);
    bool isEnd() {  return true; }
    tnode<T>* getParent();
    int getParentIndex();
    std::unordered_map<int,tnode<T>*> children;
private:
    T mapped_value;
};

template <typename T>
tnode<T>::tnode(T val, tnode<T>* /*p*/, int /*ascii*/, bool /*eow*/) {
    this->mapped_value = val;
}

template <typename T>
void tnode<T>::addChild(tnode* child, int key) {
    this->children[key] = child;
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

} // namespace trie

#endif
