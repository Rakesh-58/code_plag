import { HttpClient } from '@angular/common/http';
import { Injectable } from '@angular/core';
import { Observable } from 'rxjs';


@Injectable({
  providedIn: 'root'
})
export class PlagiarismService {

  private API_URL = 'http://localhost:5000/api/predict'; // adjust if needed

  constructor(private http: HttpClient) {}

  check(code1: string, code2: string): Observable<any> {
    return this.http.post(this.API_URL, { code1, code2 });
  }
}
