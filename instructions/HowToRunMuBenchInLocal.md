# Steps
1. Install and make sure that minikube and kubectl installed locally, and run the muBench container
```sh
docker run -it -id --platform linux/amd64 --name mubench -v msvcbench/mubench
```

2. Get the master kubernetes ip
```sh
minikube ip
```

3. Create and copy kubectl config to the muBench container
```sh
kubectl config view --flatten > config
```

4. Change the `server:<ip>:<port>` to `server:<master_ip>:8443`

5. Paste into muBench container
```sh
docker cp config mubench:/root/.kube/config
```

6. Enter muBench's bash
```sh
docker exec -it mubench bash
```

7. Check pods inside muBench
```sh
kubectl get pods -A
```
