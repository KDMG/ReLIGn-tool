# ReLIGn-tool
<p align="center">
  <img src="src/resources/logo.png" alt="Diagram" width="300"/>
</p>

The tool was presented at the [7th International Conference on Process Mining (ICPM 2025)](https://icpmconference.org/2025/awards/) in Montevideo, Uruguay, where it received the Best Demo Award 🏆.


This project provides a tool for process Model Repairing supported by Local Instance Graphs, following the approach described in the article [*Model repair supported by frequent anomalous local instance graphs*](https://www.google.com/search?client=safari&rls=en&q=model+repair+supported+by&ie=UTF-8&oe=UTF-8).
A demonstration of the tool is available at this [link](https://youtu.be/wafwsUPTees).


## Requirements
To run our application you need to have installed:
* [Python 3.9.6](https://www.python.org/downloads/release/python-396/)

* [Graphviz](https://graphviz.org/download/)

* A set of packages that you can configure inside an environment by using the [requirements.txt](https://github.com/KDMG/ReLIGn-tool/edit/main/requirements.txt) file

### Optional (only required to generate Instance Graphs)
* [Java 8](https://www.oracle.com/it/java/technologies/javase/javase8-archive-downloads.html): make sure the correct Java 8 version is in your `PATH` (`java -version` should show `1.8.x`).

* [MySQL](https://dev.mysql.com/downloads/file/?id=537130): must be configured so that the `root` user can connect **with no password**.

#### MySQL Version Requirement
This project requires the `mysql_native_password` authentication plugin.
- **MySQL 5.7** and **MySQL 8.0.x up to 8.0.34** support the plugin by default.
- **MySQL 8.4 LTS**: `mysql_native_password` is available but disabled by default.  
  To enable it, add the following to your `my.cnf` and restart MySQL:
  ```ini
  [mysqld]
  mysql_native_password=ON
- **MySQL 9.0 and above**: `mysql_native_password` has been removed and is not supported.

#### Important notes for MariaDB users

If you use **MariaDB** instead of MySQL:

1. Edit the configuration file (usually `/etc/mysql/mariadb.conf.d/50-server.cnf` on Linux).
2. Under the `[mysqld]` section, set:
   ```ini
   lower_case_table_names=1
3. Restart MariaDB:
   ```ini
   sudo systemctl restart mariadb

## Reproduce results
To run our program copy and paste the following command in your terminal:
```
git clone https://github.com/KDMG/ReLIGn-tool/
cd ReLIGn-tool

python ReLIGn.py
```
In the [`data`](https://github.com/KDMG/ReLIGn-tool/tree/main/data) folder you can find some data to test the tool.

## Contributors
| Contributor name | Contacts |
| :-------- | :------- | 
| `Claudia Diamantini`     | c.diamantini@univpm.it | 
| `Laura Genga`            | l.genga@tue.nl         | 
| `Chiara Gobbi`           | c.gobbi@pm.univpm.it   | 
| `Alessandro Mele`        | a.mele@pm.univpm.it    | 
| `Domenico Potena`        | d.potena@pm.univpm.it  | 
