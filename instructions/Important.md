# Some important stuff
1. Change the data from K8Parameters.yaml in muBench, find the ip of the `dns-resolver` and then put it in the `dns-resolver` field inside the config.
```sh
kubectl -n kube-system get svc kube-dns -o jsonpath='{.spec.clusterIP}'
```

