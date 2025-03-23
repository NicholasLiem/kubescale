# Steps
1. Install and make sure that minikube and kubectl installed locally
2. Get the master kubernetes ip
```sh
minikube ip
```
3. Create and copy kubectl config to the muBench container
```sh
kubectl config view --flatten > config
```
```sh
docker cp config mubench:/root/.kube/config
```
4. Change the `server:<ip>:<port>` to `server:<master_ip>:8443`
5. Enter muBench's bash
```sh
docker exec -it mubench bash
```
6. Check pods inside muBench
```sh
kubectl get pods -A
```
